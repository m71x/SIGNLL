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
# SAM OPTIMIZER WRAPPER
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
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

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
                    ]),
                    p=2
               )
        return norm

# =========================================================================
# STAGE 2 ONLY TRAINING LOOP
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()
    
    # 1. Model Initialization
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    # Sync initial weights across TPU cores
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    xm.rendezvous("weights_synced") 

    # --- PHASE SETUP: STAGE 2 ONLY ---
    stage_name = "STAGE 2 TEST: Halting Heads + Vectorized Entropy Gates (SAM) | Backbone FROZEN"
    
    # A. Freeze entire model first
    for param in model.parameters(): 
        param.requires_grad = False
        
    # B. Unfreeze Halting Heads (Standard List)
    for param in model.halting_heads.parameters(): 
        param.requires_grad = True

    # C. Unfreeze Entropy Gate (CRITICAL FIX: Access the single vectorized module)
    # The previous code failed here because it looked for a list 'entropy_gates'
    for param in model.entropy_gate_module.parameters(): 
        param.requires_grad = True

    if rank == 0:
        xm.master_print(f"\n{'#'*80}")
        xm.master_print(f"STARTING {stage_name}")
        xm.master_print(f"{'#'*80}")
        
        # Verify parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        xm.master_print(f"  Trainable Params: {trainable_params:,}")
        xm.master_print(f"  Frozen Params:    {frozen_params:,}")

    # --- OPTIMIZER SETUP ---
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    # Use SAM as requested for Stage 2
    optimizer = SAM(model_params, optim.AdamW, rho=0.05, adaptive=False, lr=flags["lr"], weight_decay=1e-2)
    
    # --- SCHEDULER SETUP ---
    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    total_steps = 28 * flags["epochs"] * num_batches_per_chunk
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer.base_optimizer, 
        T_0=total_steps // 4, 
        T_mult=2,   
        eta_min=1e-6
    )
    
    start_time = time.time()

    # Iterate over data chunks
    for chunk_idx in range(28): 
        current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
        
        if rank == 0:
            xm.master_print(f"Chunk {chunk_idx + 1}/28 | Loading {current_chunk_filename}")

        # Load Data (Real Data Download)
        data = training_data_download(
            core_id=rank,
            filename=current_chunk_filename,
            max_entries=total_samples
        )
        
        if data is None: 
            raise RuntimeError(f"[Core {rank}] Failed load chunk {chunk_idx}")

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        
        if teacher_cls_full.shape[1] == 25:
            teacher_cls_full = teacher_cls_full[:, 1:25, :]
        
        # Slicing for distributed training
        # Ensure divisible by num_cores for safety
        N_total_local = teacher_cls_full.shape[0]
        N_target = (N_total_local // num_cores) * num_cores

        teacher_cls_full = teacher_cls_full[:N_target]
        teacher_label_full = teacher_label_full[:N_target]

        dataset = TensorDataset(teacher_cls_full, teacher_label_full)
        sampler = RandomSampler(dataset)
        
        data_loader = DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=flags["batch_size"], 
            drop_last=True, 
            num_workers=2
        )
        
        # --- Epoch Loop ---
        for epoch in range(flags["epochs"]):
            model.train()
            
            for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                teacher_cls = teacher_cls.to(device)
                teacher_label = teacher_label.to(device)
                
                # Define Closure for SAM
                def closure():
                    # Forward Pass (Backbone is frozen, but we need the output 'z')
                    halting_logits, class_logits, z = model(teacher_cls)
                    
                    # 1. Calculate Correctness for Reweighting
                    predictions = torch.argmax(class_logits, dim=-1)
                    is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                    
                    # 2. Class-Aware Rebalancing Weights
                    n_pos = (teacher_label == 1).sum().float()
                    n_neg = (teacher_label == 0).sum().float()
                    # Prevent division by zero
                    neg_weight_val = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)
                    
                    sample_weights = torch.ones_like(halting_logits)
                    sample_weights[teacher_label == 0] = neg_weight_val.item()
                    
                    # 3. Halting Loss (Binary Cross Entropy)
                    loss_halt = F.binary_cross_entropy_with_logits(
                        halting_logits, 
                        is_correct, 
                        weight=sample_weights
                    )
                    
                    # 4. Entropy Regularization
                    # Sigmoid and clamp for numerical stability
                    h = torch.sigmoid(halting_logits).clamp(min=1e-6, max=1.0 - 1e-6)
                    
                    entropy_weight = 0.0025 
                    # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
                    h_entropy = -(h * h.log() + (1 - h) * (1 - h).log())
                    # Minimize negative entropy (maximize uncertainty slightly to prevent collapse)
                    loss_entropy = -entropy_weight * h_entropy.mean()
                    
                    loss = loss_halt + loss_entropy
                    loss.backward()
                    return loss
                
                # SAM Step
                loss = closure()
                optimizer.step(closure)
                
                # Clip Grads & Scheduler
                torch.nn.utils.clip_grad_norm_(model_params, 1.0)
                scheduler.step()
                xm.mark_step()

            # End of Epoch / Chunk Logging
            loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
            xm.rendezvous(f"chunk_{chunk_idx}_ep_{epoch}_complete")
            
            if rank == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                xm.master_print("-" * 60)
                xm.master_print(f"CHUNK {chunk_idx+1} | EPOCH {epoch+1}")
                xm.master_print(f"  LR:            {current_lr:.2e}")
                xm.master_print(f"  Stage 2 Loss:  {loss_sum / num_cores:.4f}")
                xm.master_print(f"  Elapsed Time:  {elapsed:.1f}s")

    if rank == 0:
        save_path = "controller_stage2_final.pt"
        torch.save(model.state_dict(), save_path)
        xm.master_print(f"✅ Training Complete. Saved to {save_path}")

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
        "batch_size": 32,   
        "epochs": 1, 
        "samples_per_shard": 39000
    }  
    
    print("Starting Optimized Stage 2 Training.")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')