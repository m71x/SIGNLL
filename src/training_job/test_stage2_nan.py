import os, time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from training_data_download import training_data_download

# Import base controller and utility functions
from controller_model import Controller, apply_rotary_pos_emb

# =========================================================================
# 1. DUMMY CONTROLLER (With Fixed Vectorized Gate)
# =========================================================================
class DummyEntropyController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Re-initialize the gate to ensure we use the fix below if not imported
        self.entropy_gate_module = VectorizedEntropyGate(self.d_ctrl, self.L)

    def forward(self, teacher_cls: torch.Tensor):
        B, L, D = teacher_cls.shape
        assert L == self.L

        # --- 1. Controller Body ---
        x = self.input_ln(teacher_cls)
        x = self.proj(x)
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        x = self.pre_ln(x)
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        
        z = x
        for layer in self.layers:
            z = layer(z, mask=attn_mask)
        z = self.post_ln(z) 
        
        # --- 2. Gather Class Logits ---
        class_logits_list = []
        for l in range(L):
            c_l = self.classifier_heads[l](z[:, l, :]) 
            class_logits_list.append(c_l)
        
        # --- 3. DUMMY ENTROPY INJECTION ---
        all_entropies = torch.full((B, L, 1), 0.5, device=z.device, dtype=z.dtype)
        
        # --- 4. OPTIMIZED: Run Vectorized Gate ---
        all_gated_entropies = self.entropy_gate_module(z, all_entropies)

        # --- 5. Compute Halting ---
        halting_logits_list = []
        for l in range(L):
            gated_ent_l = all_gated_entropies[:, l, :] 
            combined_input = torch.cat([z[:, l, :], gated_ent_l], dim=-1)
            h_l = self.halting_heads[l](combined_input) 
            halting_logits_list.append(h_l)
        
        halting_logits = torch.cat(halting_logits_list, dim=-1) 
        class_logits = torch.stack(class_logits_list, dim=1)
        
        return halting_logits, class_logits, z

class VectorizedEntropyGate(nn.Module):
    def __init__(self, d_ctrl: int, L: int):
        super().__init__()
        self.L = L
        self.d_ctrl = d_ctrl
        
        # Input: [B, (L*D), 1] -> Conv1d groups=L
        # Group 0 sees channels 0..D-1 (Layer 0)
        # Group 1 sees channels D..2D-1 (Layer 1)
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
        nn.init.constant_(self.net2.bias, 0.0)

    def forward(self, z, entropy):
        # z: [B, L, D]
        B, L, D = z.shape
        
        # FIX: Correct Reshape logic. 
        # Flatten L and D so that Layer 0's features are contiguous, then Layer 1, etc.
        # Old (Bugged): z.transpose(1, 2)... mixed features across layers.
        z_reshaped = z.reshape(B, L * D, 1)
        
        out = self.net1(z_reshaped)      # [B, 8*L, 1]
        out = self.act(out)
        out = self.net2(out)             # [B, L, 1]
        gate = self.sigmoid(out)
        
        gate = gate.reshape(B, L, 1)
        return entropy * gate

# =========================================================================
# 2. SAM OPTIMIZER (Float32 Precision Fix + Internal Clipping)
# =========================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, max_norm=1.0, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, max_norm=max_norm, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # 1. Clip Gradients BEFORE norm calculation
        for group in self.param_groups:
             if group.get("max_norm") is not None:
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_norm"])

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

        # 2. Clip Gradients AGAIN before final update
        for group in self.param_groups:
            if group.get("max_norm") is not None:
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_norm"])

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        # FIX: Compute Norm in Float32 to prevent bfloat16 overflow/underflow
        # The sum of squares for 100k params can easily exceed bf16 limits.
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(dtype=torch.float32, device=shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# =========================================================================
# 3. TEST LOOP
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()
    
    L = 24
    model = DummyEntropyController(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    if rank == 0: xm.master_print("Synchronizing initial weights...")
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    xm.rendezvous("weights_synced") 

    # --- PHASE SETUP: STAGE 2 ONLY ---
    stage = 2
    
    # Freeze everything first
    for param in model.parameters(): param.requires_grad = False
    
    # Unfreeze specific modules
    for param in model.halting_heads.parameters(): param.requires_grad = True
    for param in model.entropy_gate_module.parameters(): param.requires_grad = True

    if rank == 0:
        xm.master_print(f"\n{'#'*80}\nSTARTING TEST MODE: Stage 2 Only (Dummy Entropy 0.5)\n{'#'*80}")
    
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize Fixed SAM with max_norm=1.0
    optimizer = SAM(model_params, optim.AdamW, rho=0.05, adaptive=False, lr=flags["lr"], weight_decay=1e-2, max_norm=1.0)

    total_steps = 28 * flags["epochs"] * (flags["samples_per_shard"] // flags["batch_size"])
    scheduler = CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=total_steps // 4, T_mult=2, eta_min=1e-6)

    for chunk_idx in range(28): 
        current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
        if rank == 0: xm.master_print(f"Loading {current_chunk_filename}")

        data = training_data_download(core_id=rank, filename=current_chunk_filename, max_entries=flags["samples_per_shard"])
        if data is None: raise RuntimeError(f"[Core {rank}] Failed load chunk {chunk_idx}")

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        if teacher_cls_full.shape[1] == 25: teacher_cls_full = teacher_cls_full[:, 1:25, :]
        
        N_target = (teacher_cls_full.shape[0] // num_cores) * 32
        dataset = TensorDataset(teacher_cls_full[:N_target], teacher_label_full[:N_target])
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=flags["batch_size"], drop_last=True, num_workers=2)
        
        for epoch in range(flags["epochs"]):
            model.train()
            
            # Diagnostics placeholders
            diag_pos = None
            diag_neg = None

            for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                teacher_cls = teacher_cls.to(device)
                teacher_label = teacher_label.to(device)
                
                def closure():
                    halting_logits, class_logits, z = model(teacher_cls)
                    predictions = torch.argmax(class_logits, dim=-1)
                    is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                    
                    # Weights
                    n_pos = (teacher_label == 1).sum().float()
                    n_neg = (teacher_label == 0).sum().float()
                    neg_weight_val = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)
                    sample_weights = torch.ones_like(halting_logits)
                    sample_weights[teacher_label == 0] = neg_weight_val.item()
                    
                    # Loss
                    loss_halt = F.binary_cross_entropy_with_logits(halting_logits, is_correct, weight=sample_weights)
                    
                    h = torch.sigmoid(halting_logits).clamp(min=1e-6, max=1.0-1e-6)
                    h_entropy = -(h * h.log() + (1 - h) * (1 - h).log())
                    loss_entropy = -0.0025 * h_entropy.mean()
                    
                    loss = loss_halt + loss_entropy
                    loss.backward()
                    return loss
                
                # Diagnostics Capture (Pre-Update)
                if rank == 0 and batch_idx == 0:
                    with torch.no_grad():
                        h_logits, _, _ = model(teacher_cls)
                        h_probs = torch.sigmoid(h_logits)
                        
                        def get_sample(lbl):
                            mask = (teacher_label == lbl)
                            if mask.sum() > 0: return h_probs[mask][0].cpu()
                            return None
                        
                        diag_pos = get_sample(1)
                        diag_neg = get_sample(0)

                # Optimizer Step
                loss = closure()
                
                # Check for NaN gradients before step
                valid_gradients = True
                for param in model_params:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            break
                
                if valid_gradients:
                    optimizer.step(closure)
                    scheduler.step()
                else:
                    if rank == 0: print("Warning: NaN gradients detected. Skipping step.")
                    optimizer.zero_grad()
                
                xm.mark_step()

            # End of Epoch Logs
            loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
            xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")
            
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                xm.master_print("-" * 60)
                xm.master_print(f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}")
                xm.master_print(f"  LR:         {current_lr:.2e}")
                xm.master_print(f"  Total Loss: {loss_sum / num_cores:.4f}")
                
                def fmt(t, name):
                    if t is None: return f"  {name}: None"
                    return f"  > {name}: {[f'{p:.2f}' for p in t.tolist()]}"

                xm.master_print("  DIAGNOSTICS:")
                xm.master_print(fmt(diag_pos, "Sample POS"))
                xm.master_print(fmt(diag_neg, "Sample NEG"))

    xm.rendezvous("safe_exit")

def _mp_fn(rank, flags):
    try:
        torch.set_default_tensor_type('torch.FloatTensor')
        train_loop(rank, flags)
    except Exception as e:
        print(f"[Core {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "lr": 5e-4,
        "batch_size": 64,   
        "epochs": 5,
        "samples_per_shard": 39000
    }  
    print("Starting Stage 2 NaN Test (Fixed SAM + Vectorized Gate Fix).")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')