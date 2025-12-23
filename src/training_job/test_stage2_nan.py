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
# 1. DUMMY CONTROLLER (Overrides Entropy Calculation)
# =========================================================================
class DummyEntropyController(Controller):
    """
    Inherits from the standard Controller but overrides forward() 
    to inject fixed 'dummy' entropy values (0.5).
    This isolates Stage 2 testing from Stage 1 classifier weights.
    """
    def forward(self, teacher_cls: torch.Tensor):
        B, L, D = teacher_cls.shape
        assert L == self.L

        # --- 1. Controller Body (Same as Original) ---
        x = self.input_ln(teacher_cls)
        x = self.proj(x)
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        x = self.pre_ln(x)
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        
        z = x
        for layer in self.layers:
            z = layer(z, mask=attn_mask)
        z = self.post_ln(z) 
        
        # --- 2. Gather Class Logits (Standard) ---
        class_logits_list = []
        for l in range(L):
            c_l = self.classifier_heads[l](z[:, l, :]) 
            class_logits_list.append(c_l)
        
        # --- 3. DUMMY ENTROPY INJECTION ---
        # Instead of calculating entropy from untrained classifiers, 
        # we inject a fixed dummy value of 0.5 for all layers.
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

# =========================================================================
# 2. SAM OPTIMIZER (Fixed Version)
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
        # FIX 1: Clip gradients BEFORE calculating norm to prevent bfloat16 overflow (Inf)
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

        # FIX 2: Clip gradients AGAIN before the final update
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
# 3. TEST LOOP (Stage 2 Only)
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()
    
    # Init Dummy Controller
    L = 24
    model = DummyEntropyController(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    # Sync initial weights
    if rank == 0: xm.master_print("Synchronizing initial weights...")
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    xm.rendezvous("weights_synced") 

    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    start_time = time.time()

    # --- PHASE SETUP: STAGE 2 ONLY ---
    stage = 2
    stage_name = "TEST MODE: Stage 2 Only (Dummy Entropy 0.5) with Fixed SAM"
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Unfreeze halting heads
    for param in model.halting_heads.parameters(): 
        param.requires_grad = True
        
    # Unfreeze optimized entropy gate module
    for param in model.entropy_gate_module.parameters(): 
        param.requires_grad = True

    if rank == 0:
        xm.master_print(f"\n{'#'*80}")
        xm.master_print(f"STARTING {stage_name}")
        xm.master_print(f"{'#'*80}")
    
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize Fixed SAM
    optimizer = SAM(model_params, optim.AdamW, rho=0.05, adaptive=False, lr=flags["lr"], weight_decay=1e-2, max_norm=1.0)

    total_steps_in_stage = 28 * flags["epochs"] * num_batches_per_chunk
    T_0 = total_steps_in_stage // 4
    scheduler = CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)

    # Loop Chunks
    for chunk_idx in range(28): 
        current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
        
        if rank == 0:
            xm.master_print(f"Stage {stage} | Chunk {chunk_idx + 1}/28 | Loading {current_chunk_filename}")

        data = training_data_download(core_id=rank, filename=current_chunk_filename, max_entries=flags["samples_per_shard"])
        if data is None: raise RuntimeError(f"[Core {rank}] Failed load chunk {chunk_idx}")

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        
        # Dummy soft targets setup (we don't use them for Stage 2 loss, but needed for dataset)
        num_classes = 2
        t_one_hot = torch.zeros(teacher_label_full.size(0), num_classes).scatter_(1, teacher_label_full.unsqueeze(1), 1)
        teacher_log_probs_full = torch.log(t_one_hot.clamp(min=1e-10))

        if teacher_cls_full.shape[1] == 25: teacher_cls_full = teacher_cls_full[:, 1:25, :]
        
        N_target = (teacher_cls_full.shape[0] // num_cores) * 32
        dataset = TensorDataset(teacher_cls_full[:N_target], teacher_label_full[:N_target], teacher_log_probs_full[:N_target])
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=flags["batch_size"], drop_last=True, num_workers=2)
        
        for epoch in range(flags["epochs"]):
            model.train()
            diag_sample_pos = None
            diag_sample_neg = None

            for batch_idx, (teacher_cls, teacher_label, teacher_log_probs) in enumerate(data_loader):
                teacher_cls = teacher_cls.to(device)
                teacher_label = teacher_label.to(device)
                
                def closure():
                    # Forward Pass (Uses Dummy Entropy internally)
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
                
                if rank == 0 and batch_idx == 0:
                    with torch.no_grad():
                        halting_logits, class_logits, _ = model(teacher_cls)
                        # Diagnostic helper
                        def extract_sample(label_val):
                            indices = (teacher_label == label_val).nonzero(as_tuple=True)[0]
                            if indices.numel() > 0:
                                idx = indices[0]
                                return {
                                    'cls': class_logits[idx].detach().cpu(),
                                    'halt': halting_logits[idx].detach().cpu(),
                                    'lbl': teacher_label[idx].detach().cpu()
                                }
                            return None
                        diag_sample_pos = extract_sample(1)
                        diag_sample_neg = extract_sample(0)
                
                # Step
                loss = closure()
                optimizer.step(closure)
                scheduler.step()
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
                
                def format_sample(data, name):
                    if data is None: return f"  {name}: No sample found."
                    h_probs = torch.sigmoid(data['halt'])
                    return f"  > {name}:\n    HALT Probs: {[f'{p:.2f}' for p in h_probs.tolist()]}"

                xm.master_print("  DIAGNOSTICS:")
                xm.master_print(format_sample(diag_sample_pos, "Sample POS"))
                xm.master_print(format_sample(diag_sample_neg, "Sample NEG"))

        xm.rendezvous(f"chunk_end_st{stage}_ch{chunk_idx}")

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
    print("Starting Stage 2 NaN Test (Dummy Entropy).")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')