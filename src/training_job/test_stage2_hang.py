import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from controller_model import Controller
from training_data_download import training_data_download

# =========================================================================
# TRAINING LOOP 
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
    
    # Sync initial weights
    if rank == 0: xm.master_print("Synchronizing initial weights...")
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    
    xm.rendezvous("weights_synced") 

    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    global_step = 0

    # =========================================================================
    # STAGE LOOP: SKIPPING STAGE 1 (Starts directly at Stage 2)
    # =========================================================================
    for stage in [2]:
        stage_name = "TEST MODE: STAGE 2 (Halting Heads + Entropy Gates) - NO SAM, SKIPPED STAGE 1"
        
        # Setup freezing (Untrained backbone frozen, heads unfrozen)
        for param in model.parameters(): 
            param.requires_grad = False
        for param in model.halting_heads.parameters(): 
            param.requires_grad = True
        for param in model.entropy_gate_module.parameters(): 
            param.requires_grad = True

        if rank == 0:
            xm.master_print(f"\n{'#'*80}\nSTARTING {stage_name}\n{'#'*80}")
        
        # Standard Optimizer for test
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=flags["lr"], weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2, eta_min=1e-6)

        for chunk_idx in range(28): 
            current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
            if rank == 0:
                xm.master_print(f"Stage {stage} | Chunk {chunk_idx + 1}/28 | Loading {current_chunk_filename}")

            data = training_data_download(core_id=rank, filename=current_chunk_filename, max_entries=flags["samples_per_shard"])
            if data is None: raise RuntimeError(f"[Core {rank}] Failed load chunk {chunk_idx}")

            teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
            teacher_label_full = torch.from_numpy(data['classifications']).long()
            
            # Simulate Soft Targets for dummy dataset
            t_one_hot = torch.zeros(teacher_label_full.size(0), 2).scatter_(1, teacher_label_full.unsqueeze(1), 1)
            teacher_log_probs_full = torch.log(t_one_hot.clamp(min=1e-10))

            if teacher_cls_full.shape[1] == 25: 
                teacher_cls_full = teacher_cls_full[:, 1:25, :]
            
            N_target = (teacher_cls_full.shape[0] // num_cores) * 32
            
            # Dataset providing 3 values
            dataset = TensorDataset(teacher_cls_full[:N_target], teacher_label_full[:N_target], teacher_log_probs_full[:N_target])
            data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=flags["batch_size"], drop_last=True)
            
            for epoch in range(flags["epochs"]):
                model.train()
                diag_sample_pos, diag_sample_neg = None, None

                for batch_idx, (teacher_cls, teacher_label, teacher_log_probs) in enumerate(data_loader):
                    global_step += 1
                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    
                    # 1. Forward Pass
                    halting_logits, _, z = model(teacher_cls)
                    
                    # 2. CREATE DUMMY CLASS PROBABILITIES
                    # Since Stage 1 is skipped, we simulate high confidence to provide stable entropy to the heads.
                    with torch.no_grad():
                        # FIXED: halting_logits is [B, L], so we only unpack 2 values
                        B, L_dim = halting_logits.shape 
                        
                        dummy_logits = torch.zeros(B, L_dim, 2, device=device)
                        # High confidence for correct label (Logit 5.0 vs 0.0)
                        dummy_logits.scatter_(2, teacher_label.view(-1, 1, 1).expand(-1, L_dim, 1), 5.0)
                        class_logits = dummy_logits

                    is_correct = (torch.argmax(class_logits, dim=-1) == teacher_label.unsqueeze(1)).float()
                    
                    # 3. Dynamic Weighting
                    n_pos = (teacher_label == 1).sum().float()
                    n_neg = (teacher_label == 0).sum().float()
                    weights = torch.ones_like(halting_logits)
                    weights[teacher_label == 0] = (n_pos / (n_neg + 1e-6)).clamp(min=1.0).item()
                    
                    # 4. Losses
                    loss_halt = F.binary_cross_entropy_with_logits(halting_logits, is_correct, weight=weights)
                    h = torch.sigmoid(halting_logits).clamp(min=1e-6, max=1.0 - 1e-6)
                    loss_entropy = -0.0025 * -(h * h.log() + (1 - h) * (1 - h).log()).mean()
                    
                    loss = loss_halt + loss_entropy
                    
                    # 5. Diagnostics
                    if rank == 0 and batch_idx == 0:
                        with torch.no_grad():
                            # Re-run debug forward
                            h_logits_debug, _, _ = model(teacher_cls)
                            def get_diag(lbl):
                                idx = (teacher_label == lbl).nonzero(as_tuple=True)[0]
                                if idx.numel() > 0: return {'halt': h_logits_debug[idx[0]].detach().cpu(), 'lbl': lbl}
                                return None
                            diag_sample_pos, diag_sample_neg = get_diag(1), get_diag(0)
                    
                    # 6. Optimization
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    xm.mark_step()

                # Sync loss for logging
                loss_avg = xm.all_reduce(xm.REDUCE_SUM, loss) / num_cores
                xm.rendezvous(f"ep_end_st2_ch{chunk_idx}_ep{epoch}")
                
                if rank == 0:
                    xm.master_print("-" * 60)
                    xm.master_print(f"EPOCH {epoch+1} | Total Loss: {loss_avg:.4f}")
                    def fmt(d, n):
                        if not d: return f"  {n}: None"
                        return f"  > {n} (Label {d['lbl']}): HALT Probs: {[f'{p:.2f}' for p in torch.sigmoid(d['halt']).tolist()]}"
                    xm.master_print(f"  DIAGNOSTICS:\n{fmt(diag_sample_pos, 'POS')}\n{fmt(diag_sample_neg, 'NEG')}")

            xm.rendezvous(f"chunk_end_st2_ch{chunk_idx}")

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
    
    print("Starting Stage 2 Test (No SAM, Skipped Stage 1).")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')