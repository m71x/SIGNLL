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

    # Calculate step counts
    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    global_step = 0

    # =========================================================================
    # STAGE LOOP: 1 -> Backbone/Classifiers, 2 -> Halting Heads + Gates
    # =========================================================================
    for stage in [2]: # Skipping to Stage 2 for testing as requested
        
        # --- PHASE SETUP ---
        if stage == 1:
            stage_name = "STAGE 1: Backbone & Classifiers (Halting & Gates FROZEN)"
            for param in model.parameters(): 
                param.requires_grad = True
            
            # Freeze halting heads
            for param in model.halting_heads.parameters(): 
                param.requires_grad = False
            
            # Freeze optimized entropy gate module
            for param in model.entropy_gate_module.parameters(): 
                param.requires_grad = False
            
        elif stage == 2:
            stage_name = "STAGE 2: Halting Heads + Entropy Gates (Backbone FROZEN)"
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
        
        # --- OPTIMIZER SETUP ---
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        if rank == 0:
            num_params = sum(p.numel() for p in model_params)
            xm.master_print(f"Trainable parameters: {num_params:,}")
        
        optimizer = optim.AdamW(model_params, lr=flags["lr"], weight_decay=1e-2)

        # --- SCHEDULER SETUP ---
        total_steps_in_stage = 28 * flags["epochs"] * num_batches_per_chunk
        T_0 = total_steps_in_stage // 4
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0, 
            T_mult=2,   
            eta_min=1e-6
        )
        if rank == 0:
            xm.master_print(f"Scheduler: CosineAnnealingWarmRestarts, T_0={T_0} steps")

        for chunk_idx in range(28): 
            current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
            
            if rank == 0:
                xm.master_print(f"Stage {stage} | Chunk {chunk_idx + 1}/28 | Loading {current_chunk_filename}")

            # --- Load Data ---
            data = training_data_download(
                core_id=rank,
                filename=current_chunk_filename,
                max_entries=flags["samples_per_shard"]
            )
            
            if data is None: 
                raise RuntimeError(f"[Core {rank}] Failed load chunk {chunk_idx}")

            teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
            teacher_label_full = torch.from_numpy(data['classifications']).long()
            
            # --- Soft Targets ---
            if 'teacher_logits' in data:
                t_logits = torch.from_numpy(data['teacher_logits']).float()
                T_distill = 2.0
                teacher_log_probs_full = F.log_softmax(t_logits / T_distill, dim=-1)
            else:
                num_classes = 2
                smoothing = 0.1
                t_one_hot = torch.zeros(teacher_label_full.size(0), num_classes).scatter_(
                    1, teacher_label_full.unsqueeze(1), 1
                )
                teacher_probs_full = t_one_hot * (1.0 - smoothing) + (smoothing / num_classes)
                teacher_log_probs_full = torch.log(teacher_probs_full.clamp(min=1e-10))

            if teacher_cls_full.shape[1] == 25:
                teacher_cls_full = teacher_cls_full[:, 1:25, :]
            
            # --- Data Slicing ---
            N_total_local = teacher_cls_full.shape[0]
            N_target = (N_total_local // num_cores) * 32

            teacher_cls_full = teacher_cls_full[:N_target]
            teacher_label_full = teacher_label_full[:N_target]
            teacher_log_probs_full = teacher_log_probs_full[:N_target] 

            # Stage 1 Pos Weight (Static)
            neg_samples = (teacher_label_full == 0).sum().item()
            pos_samples = (teacher_label_full == 1).sum().item()
            pos_weight_val = neg_samples / (pos_samples + 1e-6)
            pos_weight_tensor = torch.tensor([pos_weight_val]).float().to(device)

            bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor).to(device)
            
            dataset = TensorDataset(teacher_cls_full, teacher_label_full, teacher_log_probs_full)
            sampler = RandomSampler(dataset)
            
            data_loader = DataLoader(
                dataset, sampler=sampler, batch_size=flags["batch_size"], 
                drop_last=True, num_workers=2
            )
            
            # --- Epoch Loop ---
            for epoch in range(flags["epochs"]):
                model.train()
                
                # FIX 1: Use running averages instead of accumulating tensors
                running_loss = 0.0
                num_batches = 0
                
                # FIX 2: Store only scalar values for diagnostics
                diag_halt_pos = None
                diag_halt_neg = None

                for batch_idx, (teacher_cls, teacher_label, teacher_log_probs) in enumerate(data_loader):
                    if stage == 2: 
                        global_step += 1
                    
                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    teacher_log_probs = teacher_log_probs.to(device)
                    
                    # --- STAGE 1: Standard Training ---
                    if stage == 1:
                        halting_logits, class_logits, z = model(teacher_cls) 
                        
                        # --- LOSS CALCULATION ---
                        labels = teacher_label.float().unsqueeze(1).expand(-1, L)
                        
                        if class_logits.size(-1) == 2:
                            class_logits_positive = class_logits[:, :, 1]
                            student_log_probs = F.log_softmax(class_logits, dim=-1)
                        else:
                            class_logits_positive = class_logits.squeeze(-1)
                            student_log_probs = F.log_softmax(
                                torch.stack([-class_logits_positive, class_logits_positive], dim=-1),
                                dim=-1
                            )
                        
                        loss_hard = bce_loss_fn(class_logits_positive, labels)

                        teacher_log_probs_expanded = teacher_log_probs.unsqueeze(1).expand(-1, L, -1)
                        kl_elementwise = teacher_log_probs_expanded.exp() * (
                            teacher_log_probs_expanded - student_log_probs
                        )
                        loss_soft = kl_elementwise.sum(dim=-1)

                        alpha = 0.5
                        ce_per_layer = (alpha * loss_hard) + ((1 - alpha) * loss_soft)

                        loss_contrast = F.mse_loss(z[:, 1:, :], z[:, :-1, :])

                        loss_cls = ce_per_layer.mean()
                        loss_halt = torch.tensor(0.0, device=device)
                        loss = (loss_cls * 2) + (0.1 * loss_contrast)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        xm.optimizer_step(optimizer)
                        scheduler.step()
                        
                        # FIX 3: CRITICAL - mark_step after EVERY batch to prevent graph accumulation
                        xm.mark_step()
                        
                        # FIX 4: Update running average with detached scalar
                        running_loss += loss.detach().item()
                        num_batches += 1

                    # --- STAGE 2: Standard Training ---
                    elif stage == 2:
                        # Forward pass
                        halting_logits, class_logits, z = model(teacher_cls)
                        
                        predictions = torch.argmax(class_logits, dim=-1)
                        is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                        
                        # CLASS-AWARE REBALANCING (OPTIMIZED)
                        n_pos = (teacher_label == 1).sum().float()
                        n_neg = (teacher_label == 0).sum().float()
                        neg_weight_tensor = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)
                        
                        mask_neg = (teacher_label == 0).unsqueeze(1) 
                        
                        sample_weights = torch.ones_like(halting_logits)
                        sample_weights = torch.where(mask_neg, neg_weight_tensor, sample_weights)
                        
                        # Halting Loss
                        loss_halt = F.binary_cross_entropy_with_logits(
                            halting_logits, 
                            is_correct, 
                            weight=sample_weights
                        )
                        
                        # Stable entropy calculation
                        h = torch.sigmoid(halting_logits)
                        h_safe = h.clamp(min=1e-7, max=1.0 - 1e-7)
                        
                        entropy_weight = 0.0025 
                        h_entropy = -(h_safe * h_safe.log() + (1 - h_safe) * (1 - h_safe).log())
                        h_entropy = torch.nan_to_num(h_entropy, nan=0.0, posinf=0.0, neginf=0.0)
                        loss_entropy = -entropy_weight * h_entropy.mean()
                        
                        loss = loss_halt + loss_entropy
                        
                        # Safety check before backward
                        if not torch.isfinite(loss):
                            if rank == 0:
                                xm.master_print(f"WARNING: Non-finite loss at batch {batch_idx}, skipping")
                            continue
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        xm.optimizer_step(optimizer)
                        scheduler.step()
                        
                        # FIX 5: CRITICAL - mark_step after EVERY batch
                        xm.mark_step()
                        
                        # FIX 6: Update running average with detached scalar
                        running_loss += loss.detach().item()
                        num_batches += 1
                        
                        # FIX 7: Extract diagnostics ONLY on first batch, convert to scalars immediately
                        if rank == 0 and batch_idx == 0:
                            with torch.no_grad():
                                h_probs = torch.sigmoid(halting_logits)
                                pos_mask = teacher_label == 1
                                neg_mask = teacher_label == 0
                                
                                if pos_mask.any():
                                    # Store as Python list of floats (not tensors)
                                    diag_halt_pos = h_probs[pos_mask][0].cpu().tolist()
                                if neg_mask.any():
                                    diag_halt_neg = h_probs[neg_mask][0].cpu().tolist()

                # FIX 8: Compute average loss (already scalar, no graph)
                avg_loss = running_loss / max(num_batches, 1)
                
                # FIX 9: Create scalar tensor for reduction (no graph accumulation)
                loss_tensor = torch.tensor(avg_loss, device=device)
                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss_tensor)
                
                # FIX 10: mark_step BEFORE rendezvous
                xm.mark_step()
                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")
                
                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    xm.master_print("-" * 60)
                    xm.master_print(f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}")
                    xm.master_print(f"  LR:         {current_lr:.2e}")
                    xm.master_print(f"  Avg Loss:   {loss_sum.item() / num_cores:.4f}")
                    
                    # Print diagnostics if available (already Python lists)
                    if stage == 2:
                        if diag_halt_pos:
                            xm.master_print(f"  Sample POS Halt: {[f'{p:.2f}' for p in diag_halt_pos]}")
                        if diag_halt_neg:
                            xm.master_print(f"  Sample NEG Halt: {[f'{p:.2f}' for p in diag_halt_neg]}")

            xm.mark_step()
            xm.rendezvous(f"chunk_end_st{stage}_ch{chunk_idx}")

    xm.rendezvous("ready_to_save_final")
    save_path = os.path.expanduser("~/SIGNLL/final_model_stage2_gated.pt")
    
    if rank == 0:
        xm.master_print(f"Saving final model: {save_path}")
        torch.save(model.state_dict(), save_path)
        xm.master_print("✅ Training Complete with Gated Entropy.")

    xm.rendezvous("save_complete_safe_exit")

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
    
    print("Starting Two-Stage Training with Gated Entropy (TEST MODE: STAGE 2 ONLY).")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')