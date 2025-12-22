import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from controller_model import Controller
from training_data_download import training_data_download

# =========================================================================
# SAM OPTIMIZER WRAPPER (Unchanged)
# =========================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
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
        assert closure is not None, "SAM requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]), p=2)
        return norm

# =========================================================================
# STAGE 2 ONLY TRAINING LOOP
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()
    
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    # Sync initial weights
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    xm.rendezvous("weights_synced") 

    # --- PHASE SETUP: STAGE 2 ONLY ---
    stage_name = "STAGE 2 TEST: Halting Heads + Entropy Gates (SAM) | Backbone FROZEN"
    
    # Freeze everything first
    for param in model.parameters(): 
        param.requires_grad = False
        
    # Unfreeze halting heads AND entropy gates for Stage 2
    for param in model.halting_heads.parameters(): 
        param.requires_grad = True
    for param in model.entropy_gates.parameters(): 
        param.requires_grad = True

    if rank == 0:
        xm.master_print(f"\n{'#'*80}\nSTARTING {stage_name}\n{'#'*80}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        xm.master_print(f"Trainable Stage 2 parameters: {num_params:,}")

    # Optimizer & Scheduler
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SAM(model_params, optim.AdamW, rho=0.05, lr=flags["lr"], weight_decay=1e-2)
    
    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    total_steps = 28 * flags["epochs"] * num_batches_per_chunk
    
    scheduler = CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=total_steps // 4, T_mult=2, eta_min=1e-6)
    
    start_time = time.time()

    for chunk_idx in range(28): 
        current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
        if rank == 0: xm.master_print(f"Loading {current_chunk_filename}...")

        data = training_data_download(core_id=rank, filename=current_chunk_filename, max_entries=total_samples)
        if data is None: raise RuntimeError(f"Failed load chunk {chunk_idx}")

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        
        if teacher_cls_full.shape[1] == 25:
            teacher_cls_full = teacher_cls_full[:, 1:25, :]
        
        # Slicing
        N_target = (teacher_cls_full.shape[0] // num_cores) * num_cores
        dataset = TensorDataset(teacher_cls_full[:N_target], teacher_label_full[:N_target])
        data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=flags["batch_size"], drop_last=True)

        for epoch in range(flags["epochs"]):
            model.train()
            
            for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                teacher_cls, teacher_label = teacher_cls.to(device), teacher_label.to(device)
                
                def closure():
                    # We only care about halting and correctness (Stage 2)
                    halting_logits, class_logits, _ = model(teacher_cls)
                    
                    # Compute correctness for rebalancing
                    predictions = torch.argmax(class_logits, dim=-1)
                    is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                    
                    # Class-aware weights
                    n_pos = (teacher_label == 1).sum().float()
                    n_neg = (teacher_label == 0).sum().float()
                    neg_weight_val = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)
                    
                    sample_weights = torch.ones_like(halting_logits)
                    sample_weights[teacher_label == 0] = neg_weight_val.item()
                    
                    loss_halt = F.binary_cross_entropy_with_logits(halting_logits, is_correct, weight=sample_weights)
                    
                    # Entropy regularization
                    h = torch.sigmoid(halting_logits).clamp(min=1e-6, max=1.0 - 1e-6)
                    loss_entropy = -0.0025 * (-(h * h.log() + (1 - h) * (1 - h).log())).mean()
                    
                    total_loss = loss_halt + loss_entropy
                    total_loss.backward()
                    return total_loss
                
                loss = closure()
                optimizer.step(closure)
                scheduler.step()
                xm.mark_step()

            if rank == 0:
                elapsed = time.time() - start_time
                xm.master_print(f"Chunk {chunk_idx+1} | Epoch {epoch+1} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")

    if rank == 0:
        xm.master_print("✅ Stage 2 Test Complete.")

def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    train_loop(rank, flags)

if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "lr": 5e-4,
        "batch_size": 32,   
        "epochs": 1, # Set to 1 for quick speed testing
        "samples_per_shard": 10000 # Reduced for the speed test
    }  
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')