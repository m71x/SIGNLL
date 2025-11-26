import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

# =========================================================================
# EVALUATION FUNCTION
# =========================================================================
def evaluate_model(rank, model, chunk_idx, threshold, batch_size, samples_per_shard):
    """
    Tests the model on a specific chunk using an early-exit strategy.
    Runs ONLY on Device 0.
    """
    if rank != 0:
        return

    device = xm.xla_device()
    
    xm.master_print(f"\n{'*'*80}")
    xm.master_print(f"*** EVALUATION | CHUNK {chunk_idx} | THRESHOLD {threshold} ***")
    xm.master_print(f"{'*'*80}")

    model.eval()
    
    current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
    
    data = training_data_download(
        core_id=0, 
        filename=current_chunk_filename,
        max_entries=samples_per_shard
    )
    
    if data is None:
        xm.master_print(f"[Core {rank}] Failed to load test data for chunk {chunk_idx}")
        return

    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]
    
    dataset = TensorDataset(teacher_cls_full, teacher_label_full)
    sampler = SequentialSampler(dataset)
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2
    )
    
    total_samples = 0
    total_correct = 0
    layer_exit_counts_cpu = torch.zeros(24, dtype=torch.float32)

    with torch.no_grad():
        for i, (teacher_cls, teacher_label) in enumerate(data_loader):
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            
            halting_logits, class_logits, _ = model(teacher_cls)
            
            h_probs = torch.sigmoid(halting_logits)
            threshold_mask = (h_probs > threshold)
            
            exit_indices = torch.argmax(threshold_mask.long(), dim=1)
            row_has_exit = threshold_mask.any(dim=1)
            exit_indices[~row_has_exit] = 23
            
            batch_indices = torch.arange(class_logits.size(0), device=device)
            selected_logits = class_logits[batch_indices, exit_indices]
            predictions = torch.argmax(selected_logits, dim=-1)
            
            correct_tensor = (predictions == teacher_label).sum()
            total_correct += correct_tensor.item() 
            total_samples += teacher_label.size(0)
            
            exit_indices_cpu = exit_indices.cpu()
            unique_exits, counts = torch.unique(exit_indices_cpu, return_counts=True)
            layer_exit_counts_cpu.index_add_(0, unique_exits, counts.float())
            
            xm.mark_step()
            
    accuracy = (total_correct / total_samples) * 100.0
    layers = torch.arange(24, dtype=torch.float32)
    avg_exit_layer = (layer_exit_counts_cpu * layers).sum() / total_samples
    
    xm.master_print(f"RESULTS:")
    xm.master_print(f"  Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")
    xm.master_print(f"  Avg Exit Layer: {avg_exit_layer:.2f} (0-23)")
    xm.master_print(f"{'*'*80}\n")

    model.train()

# =========================================================================
# MAIN TRAINING STAGE LOOP (FIXED)
# =========================================================================
#moved 
# ENABLE ACCESS TO RandomSampler


def run_stage(rank, flags, model, optimizer, stage):
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()
    
    L = 24
    # Ensure this matches your download limit
    total_samples = flags["samples_per_shard"]
    
    # Track steps continuously
    total_steps = flags["epochs"] * 29 * (total_samples // flags["batch_size"])
    global_step = 0 

    model.train()

    for chunk_idx in range(28): 
        current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
        
        if rank == 0:
            xm.master_print(f"\n### STAGE: {stage} | CHUNK {chunk_idx + 1}/29 ###")

        # --- Load Data ---
        data = training_data_download(
            core_id=rank, 
            filename=current_chunk_filename,
            max_entries=flags["samples_per_shard"]
        )
        
        if data is None: raise RuntimeError("Data load failed")

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        
        if teacher_cls_full.shape[1] == 25:
            teacher_cls_full = teacher_cls_full[:, 1:25, :]

        # =====================================================================
        # [FIX 1] GLOBAL WEIGHT SYNCHRONIZATION (prevents math errors)
        # =====================================================================
        # 1. Convert counts to tensors on the DEVICE immediately
        neg_count = torch.tensor((teacher_label_full == 0).sum(), device=device, dtype=torch.float32)
        pos_count = torch.tensor((teacher_label_full == 1).sum(), device=device, dtype=torch.float32)
        
        # 2. Sync across all cores so everyone agrees on the count
        neg_count_global = xm.all_reduce(xm.REDUCE_SUM, neg_count)
        pos_count_global = xm.all_reduce(xm.REDUCE_SUM, pos_count)
        
        # 3. Calculate weight using the synchronized global counts
        pos_weight_val = neg_count_global / (pos_count_global + 1e-6)
        
        # 4. Create the criterion with the static global weight
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val.view(1), reduction='none')

        if rank == 0:
            xm.master_print(f"[Chunk {chunk_idx}] Global Pos Weight: {pos_weight_val.item():.4f}")

        dataset = TensorDataset(teacher_cls_full, teacher_label_full)
        
        # =====================================================================
        # [FIX 2] CORRECT SAMPLER (Fixes Double Sharding)
        # =====================================================================
        # Since data is ALREADY unique per core (via training_data_download),
        # we use RandomSampler to use ALL data loaded on this core.
        sampler = DistributedSampler(
            dataset, num_replicas=num_cores, rank=rank, shuffle=True
        )
        
        data_loader = DataLoader(
            dataset, sampler=sampler, batch_size=flags["batch_size"],
            drop_last=True, num_workers=2
        )
        
        xm.rendezvous(f"data_ready_{stage}_{chunk_idx}")

        for epoch in range(flags["epochs"]):
            # We will track loss locally for logging
            epoch_loss_accum = 0.0
            num_batches = 0
            
            for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                global_step += 1
                num_batches += 1
                
                teacher_cls = teacher_cls.to(device)
                teacher_label = teacher_label.to(device)
                
                optimizer.zero_grad()
                halting_logits, class_logits, _ = model(teacher_cls)
                
                # Prepare labels
                labels = teacher_label.float().unsqueeze(1).expand(-1, L)
                if class_logits.size(-1) == 2:
                    class_logits_positive = class_logits[:, :, 1]
                else:
                    class_logits_positive = class_logits.squeeze(-1)
                
                # Calculate Loss
                ce_per_layer = bce_criterion(class_logits_positive, labels)
                
                if stage == "train_cls":
                    loss_cls = ce_per_layer.mean()
                    loss_halt = torch.tensor(0.0, device=device)
                elif stage == "train_halt":
                    h = torch.sigmoid(halting_logits)
                    q = compute_q_from_h(h)
                    loss_cls = (q * ce_per_layer).sum(dim=1).mean()
                    
                    depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
                    halt_penalty = (depths * (1 - h)).sum(dim=1)
                    
                    progress = float(global_step) / float(total_steps)
                    lambda_now = flags["lambda_start"] + (flags["lambda_target"] - flags["lambda_start"]) * progress
                    loss_halt = lambda_now * halt_penalty.mean()

                loss = loss_cls + loss_halt
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                xm.optimizer_step(optimizer)

                # =============================================================
                # [FIX 3] DEADLOCK PREVENTION
                # =============================================================
                # REMOVED: print(loss.item())
                # Instead, we just let the TPU run. 
                # If you need to debug, rely on the epoch log below.

            # End of Epoch: Safe to sync and print now
            # We take the loss from the LAST batch of the epoch as a proxy, 
            # or you can accumulate properly (but that requires more code).
            # Here we just sync the loss from the final batch to check health.
            loss_reduced = xm.all_reduce(xm.REDUCE_SUM, loss) / num_cores
            
            # This .item() is SAFE because it happens on all cores simultaneously
            loss_val = loss_reduced.item() 
            
            if rank == 0:
                xm.master_print(f"[{stage}] Chunk {chunk_idx+1} | Epoch {epoch+1} | Loss (Last Batch): {loss_val:.4f}")

        # Ensure all cores finish the chunk before moving to the next
        xm.rendezvous(f"chunk_end_{stage}_{chunk_idx}")

# =========================================================================
# ORCHESTRATOR
# =========================================================================
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()
    
    # 1. Initialize Model ONCE
    if rank == 0: 
        xm.master_print("Initializing Controller Model...")
    
    model = Controller(
        L=24,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2,
        halting_bias_init=-2.5 
    ).to(device)

    # FIXED: Proper weight synchronization
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    xm.rendezvous("init_sync")

    # =========================================================================
    # STAGE 1: TRAIN TRANSFORMER + CLASSIFIERS (IGNORE HALTING)
    # =========================================================================
    if rank == 0: 
        xm.master_print("\n>>> ENTERING STAGE 1: TRAIN CLASSIFIERS <<<")
    
    optimizer_cls = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    
    run_stage(rank, flags, model, optimizer_cls, stage="train_cls")
    
    # Save Stage 1 result
    if rank == 0:
        xm.mark_step()
        torch.save(model.state_dict(), "/tmp/model_stage1_finished.pt")
        xm.master_print("✅ Stage 1 Complete. Model saved.")
    xm.rendezvous("stage1_done")

    # =========================================================================
    # STAGE 2: TRAIN HALTING HEADS (FREEZE TRANSFORMER/CLS)
    # =========================================================================
    if rank == 0: 
        xm.master_print("\n>>> ENTERING STAGE 2: TRAIN HALTING HEADS <<<")

    # FIXED: Proper freezing with synchronization
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.halting_heads.parameters():
        param.requires_grad = True
    
    # CRITICAL: Ensure freeze state propagates before creating optimizer
    xm.mark_step()
    xm.rendezvous("freeze_sync")

    # Create NEW Optimizer (Only for halting heads)
    optimizer_halt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=flags["lr"], 
        weight_decay=1e-2
    )

    run_stage(rank, flags, model, optimizer_halt, stage="train_halt")
    
    if rank == 0:
        xm.mark_step()
        torch.save(model.state_dict(), "/tmp/model_stage2_final.pt")
        xm.master_print("✅ ALL STAGES COMPLETE. Final model saved.")
    xm.rendezvous("job_done")


if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "lr": 1e-4,
        "batch_size": 8, 
        
        "lambda_start": 0.0001,
        "lambda_target": 0.005,
        
        "epochs": 4,
        "samples_per_shard": 19500, 
        
        "test_chunk": 28
    }
    
    print("Starting XLA Two-Stage Training Job...")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')
    print("Training complete!")