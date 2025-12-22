import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 

# =========================================================================
# 1. OPTIMIZED MODEL (Vectorized Gates)
# =========================================================================

class OptimizedEntropyGate(nn.Module):
    """
    Replaces the list of 24 separate EntropyGates with a single Grouped Convolution.
    This maintains UNIQUE weights per layer but runs as 1 fast operation.
    """
    def __init__(self, d_ctrl: int, L: int):
        super().__init__()
        self.L = L
        # We use Conv1d with groups=L to simulate L independent Linear layers
        # Input: [B, D*L, 1] (reshaped) -> Output: [B, 8*L, 1]
        self.net1 = nn.Conv1d(
            in_channels=d_ctrl * L, 
            out_channels=8 * L, 
            kernel_size=1, 
            groups=L
        )
        self.act = nn.Tanh()
        self.net2 = nn.Conv1d(
            in_channels=8 * L, 
            out_channels=1 * L, 
            kernel_size=1, 
            groups=L
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, entropy):
        # z: [B, L, D]
        # entropy: [B, L, 1]
        B, L, D = z.shape
        
        # 1. Permute for Conv1d: [B, D*L, 1]
        # We group by L so Conv1d sees (Layer1_Feats, Layer2_Feats, ...)
        z_reshaped = z.permute(0, 2, 1).reshape(B, D * L, 1)
        
        # 2. Forward Pass (Vectorized)
        out = self.net1(z_reshaped)      # [B, 8*L, 1]
        out = self.act(out)
        out = self.net2(out)             # [B, L, 1]
        gate = self.sigmoid(out)
        
        # 3. Reshape back to [B, L, 1]
        gate = gate.reshape(B, L, 1)
        
        return entropy * gate

class OptimizedController(nn.Module):
    def __init__(self, L=24, d_teacher=1024, d_ctrl=512, n_layers=4, num_classes=2):
        super().__init__()
        self.L = L
        self.d_ctrl = d_ctrl
        self.num_classes = num_classes
        
        # Mocking the Backbone for speed test (just projections)
        self.proj = nn.Linear(d_teacher, d_ctrl)
        self.layers = nn.ModuleList([
            nn.Linear(d_ctrl, d_ctrl) for _ in range(n_layers) # Simplified transformer
        ])
        
        # --- THE FIX: Single Vectorized Module instead of List ---
        self.vectorized_gate = OptimizedEntropyGate(d_ctrl, L)
        
        # Halting Heads (We can leave these as List for now, or vectorize similarly)
        self.halting_heads = nn.ModuleList([
            nn.Linear(d_ctrl + 1, 1) for _ in range(L)
        ])
        self.classifier_heads = nn.ModuleList([
            nn.Linear(d_ctrl, num_classes) for _ in range(L)
        ])

    def forward(self, teacher_cls):
        # Simplified Forward for Speed Test
        B, L, D = teacher_cls.shape
        x = self.proj(teacher_cls) # [B, L, d_ctrl]
        
        # Fake Transformer blocks
        for layer in self.layers:
            x = x + layer(x)
        z = x
        
        # 1. Compute Class Logits & Entropy for ALL layers at once
        class_logits_list = []
        for l in range(L):
            class_logits_list.append(self.classifier_heads[l](z[:, l, :]))
        class_logits = torch.stack(class_logits_list, dim=1) # [B, L, 2]
        
        # Calculate Entropy
        log_probs = F.log_softmax(class_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True) # [B, L, 1]
        
        # 2. VECTORIZED GATE (The Speed Fix)
        # Instead of looping 24 times, we run once
        gated_entropy = self.vectorized_gate(z, entropy) # [B, L, 1]
        
        # 3. Halting Heads
        halting_logits_list = []
        for l in range(L):
            # Input is hidden state + the gated entropy for this layer
            combined = torch.cat([z[:, l, :], gated_entropy[:, l, :]], dim=-1)
            halting_logits_list.append(self.halting_heads[l](combined))
            
        halting_logits = torch.cat(halting_logits_list, dim=-1)
        
        return halting_logits, class_logits, z

# =========================================================================
# 2. SAM OPTIMIZER (Same as yours)
# =========================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # Optimization: Move norm calculation to XLA more efficiently if possible
        # but sticking to your implementation for consistency
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2
        )
        return norm

# =========================================================================
# 3. DUMMY TRAINING LOOP (Stage 2 Only)
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    
    # Init Optimized Model
    model = OptimizedController(d_ctrl=flags['d_ctrl']).to(device)
    
    # Mock syncing weights
    xm.mark_step()
    
    # FREEZE BACKBONE (Stage 2 setup)
    for p in model.parameters(): p.requires_grad = False
    for p in model.halting_heads.parameters(): p.requires_grad = True
    for p in model.vectorized_gate.parameters(): p.requires_grad = True
    
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SAM(model_params, optim.AdamW, lr=0.001)
    
    # Synthetic Data Generator
    batch_size = flags['batch_size']
    L = 24
    d_teacher = 1024
    
    xm.master_print(">>> Starting Dummy Stage 2 Training (Speed Test)")
    xm.master_print(f">>> Batch Size: {batch_size}, Optimization: SAM")
    
    start_time = time.time()
    
    # Run 50 steps
    for step in range(50):
        # Generate random dummy data on device
        # We use fixed class probabilities effectively by having random inputs 
        # that don't change much, but the model will process them fully.
        teacher_cls = torch.randn(batch_size, L, d_teacher, device=device)
        teacher_label = torch.randint(0, 2, (batch_size,), device=device)
        
        def closure():
            halting_logits, class_logits, z = model(teacher_cls)
            
            # Simple Loss
            loss = halting_logits.mean() + class_logits.mean() # Dummy loss
            loss.backward()
            return loss
        
        loss = closure()
        optimizer.step(closure)
        xm.mark_step()
        
        if step % 10 == 0 and rank == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}: {elapsed:.2f}s elapsed ({(elapsed/(step+1)):.3f} s/step)")

    if rank == 0:
        total_time = time.time() - start_time
        print(f"✅ Finished 50 steps in {total_time:.2f}s")
        print(f"✅ Avg time per step: {total_time/50:.4f}s")

def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    train_loop(rank, flags)

if __name__ == "__main__":
    FLAGS = {"d_ctrl": 512, "batch_size": 32}
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method='fork')