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

# Import base controller and utility functions from your uploaded module
from controller_model import Controller, apply_rotary_pos_emb

# =========================================================================
# 1. FIXED VECTORIZED GATE
# =========================================================================
class VectorizedEntropyGate(nn.Module):
    def __init__(self, d_ctrl: int, L: int):
        super().__init__()
        self.L = L
        self.d_ctrl = d_ctrl
        self.net1 = nn.Conv1d(in_channels=d_ctrl * L, out_channels=8 * L, kernel_size=1, groups=L)
        self.act = nn.Tanh()
        self.net2 = nn.Conv1d(in_channels=8 * L, out_channels=1 * L, kernel_size=1, groups=L)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.net2.bias, 0.0)

    def forward(self, z, entropy):
        B, L, D = z.shape
        # Ensure Layer features stay contiguous for Grouped Conv1d
        z_reshaped = z.reshape(B, L * D, 1)
        out = self.net1(z_reshaped)
        out = self.act(out)
        out = self.net2(out)
        gate = self.sigmoid(out)
        return entropy * gate.reshape(B, L, 1)

# =========================================================================
# 2. DUMMY CONTROLLER (Graph-Optimized)
# =========================================================================
class DummyEntropyController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entropy_gate_module = VectorizedEntropyGate(self.d_ctrl, self.L)

    def forward(self, teacher_cls: torch.Tensor):
        B, L, D = teacher_cls.shape
        x = self.input_ln(teacher_cls)
        x = self.proj(x)
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        x = self.pre_ln(x)
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        
        z = x
        for layer in self.layers:
            z = layer(z, mask=attn_mask)
        z = self.post_ln(z) 
        
        class_logits = torch.stack([self.classifier_heads[l](z[:, l, :]) for l in range(L)], dim=1)
        all_entropies = torch.full((B, L, 1), 0.5, device=z.device, dtype=z.dtype)
        all_gated_entropies = self.entropy_gate_module(z, all_entropies)

        halting_logits = torch.cat([
            self.halting_heads[l](torch.cat([z[:, l, :], all_gated_entropies[:, l, :]], dim=-1)) 
            for l in range(L)
        ], dim=-1)
        
        return halting_logits, class_logits, z

# =========================================================================
# 3. ROBUST SAM OPTIMIZER (Float32 Norms)
# =========================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, max_norm=1.0, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, max_norm=max_norm, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # Clip BEFORE norm to prevent bf16 overflow
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
        # Clip BEFORE update
        for group in self.param_groups:
            if group.get("max_norm") is not None:
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_norm"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        # Force float32 for the norm reduction to prevent bfloat16 SumOfSquares overflow
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(dtype=torch.float32, device=shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]), p=2)
        return norm

# =========================================================================
# 4. TRAINING LOOP (Single Forward Pass Optimization)
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    model = DummyEntropyController(d_ctrl=flags["d_ctrl"], n_layers=flags["transformer_layers"]).to(device)
    
    # Freeze logic
    for p in model.parameters(): p.requires_grad = False
    for p in model.halting_heads.parameters(): p.requires_grad = True
    for p in model.entropy_gate_module.parameters(): p.requires_grad = True

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SAM(model_params, optim.AdamW, lr=flags["lr"], max_norm=1.0)
    
    # Data loading
    data = training_data_download(core_id=rank, filename="embeddings_chunk_0.npz", max_entries=flags["samples_per_shard"])
    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()[:, 1:25, :]
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    dataset = TensorDataset(teacher_cls_full, teacher_label_full)
    data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=flags["batch_size"], drop_last=True)

    xm.master_print(f"\nSTARTING STAGE 2 NaN TEST (Batch Size: {flags['batch_size']})")
    
    for epoch in range(flags["epochs"]):
        model.train()
        diag_probs = None # To store diagnostic info from inside the closure

        for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
            teacher_cls, teacher_label = teacher_cls.to(device), teacher_label.to(device)

            def closure():
                nonlocal diag_probs
                halting_logits, class_logits, _ = model(teacher_cls)
                is_correct = (torch.argmax(class_logits, dim=-1) == teacher_label.unsqueeze(1)).float()
                
                # Loss calculation
                loss = F.binary_cross_entropy_with_logits(halting_logits, is_correct)
                h = torch.sigmoid(halting_logits).clamp(1e-6, 1.0-1e-6)
                loss += -0.0025 * (-(h * h.log() + (1-h) * (1-h).log())).mean()
                
                loss.backward()
                
                # Capture diagnostics ONLY on first batch of first core to avoid graph overhead
                if rank == 0 and batch_idx == 0:
                    diag_probs = h.detach().cpu()[0] 
                return loss

            # SAM Step
            optimizer.step(closure)
            xm.mark_step() # Force graph execution here

            if batch_idx % 20 == 0 and rank == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} processed.")

        if rank == 0:
            xm.master_print(f"Epoch {epoch} Complete. Diagnostic Halt Probs (Sample 0): {[f'{p:.2f}' for p in diag_probs.tolist()]}")

def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    train_loop(rank, flags)

if __name__ == "__main__":
    BASE_FLAGS = {"d_ctrl": 512, "transformer_layers": 4, "lr": 5e-4, "batch_size": 64, "epochs": 2, "samples_per_shard": 5000}
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')