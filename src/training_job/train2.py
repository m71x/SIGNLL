import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# (1) Import new classes
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# NOTE: Assuming these imports resolve correctly in the environment
from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

# DO NOT USE .item(), IT WILL FORCE XLA TO RECOMPILE AT EVERY BATCH ITERATION

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
    
    # Force loading for core 0
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
    
    # Handle layer slicing
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
            
            # --- Early Exit Logic ---
            h_probs = torch.sigmoid(halting_logits)
            threshold_mask = (h_probs > threshold)
            
            exit_indices = torch.argmax(threshold_mask.long(), dim=1)
            row_has_exit = threshold_mask.any(dim=1)
            exit_indices[~row_has_exit] = 23
            
            # --- Classification ---
            batch_indices = torch.arange(class_logits.size(0), device=device)
            selected_logits = class_logits[batch_indices, exit_indices]
            predictions = torch.argmax(selected_logits, dim=-1)
            
            # --- Metrics ---
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
# MAIN TRAINING STAGE LOOP
# =========================================================================
def run_stage(rank, flags, model, optimizer, stage):
    """
    Executes a full pass over all 29 chunks for a specific stage (CLS or HALT).
    """
    device = xm.torch_xla.device()
    num_cores = xm.xrt_world_size()
    
    L = 24
    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    total_steps = flags["epochs"] * 29 * num_batches_per_chunk
    
    global_step = 0
    start_time = time.time()
    model.train()

    if rank == 0:
        xm.master_print(f"\n{'='*80}")
        xm.master_print(f"STARTING STAGE: {stage.upper()}")
        xm.master_print(f"{'='*80}")

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
        
        if data is None:
            raise RuntimeError(f"Failed to load data for chunk {chunk_idx}")

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        
        if teacher_cls_full.shape[1] == 25:
            teacher_cls_full = teacher_cls_full[:, 1:25, :]

        # --- Class Weighting ---
        neg_samples_count = (teacher_label_full == 0).sum().item()
        pos_samples_count = (teacher_label_full == 1).sum().item()
        pos_weight_value = neg_samples_count / pos_samples_count
        pos_weight_tensor = torch.tensor([pos_weight_value]).float().to(device)

        bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor).to(device)

        # --- DataLoader ---
        dataset = TensorDataset(teacher_cls_full, teacher_label_full)
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_cores,
            rank=rank,
            shuffle=True
        )
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=flags["batch_size"],
            drop_last=True,
            num_workers=2
        )
        
        xm.rendezvous(f"data_ready_{stage}_{chunk_idx}")

        # --- Epoch Loop ---
        for epoch in range(flags["epochs"]):
            
            for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                global_step += 29
                teacher_cls = teacher_cls.to(device)
                teacher_label = teacher_label.to(device)
                
                # Forward
                halting_logits, class_logits, _ = model(teacher_cls)
                
                # Loss Calculation
                labels = teacher_label.float().unsqueeze(1).expand(-1, L)
                if class_logits.size(-1) == 2:
                    class_logits_positive = class_logits[:, :, 1]
                else:
                    class_logits_positive = class_logits.squeeze(-1)
                
                ce_per_layer = bce_loss_fn(class_logits_positive, labels)
                
                if stage == "train_cls":
                    # STAGE 1: Train ALL classifiers. Ignore halting.
                    loss_cls = ce_per_layer.mean()
                    loss_halt = torch.tensor(0.0, device=device)
                    
                elif stage == "train_halt":
                    # STAGE 2: Train Halting Heads using frozen classifier confidence
                    h = torch.sigmoid(halting_logits)
                    q = compute_q_from_h(h)
                    
                    # Weighted loss: Halting head learns to assign high q where error is low
                    loss_cls = (q * ce_per_layer).sum(dim=1).mean()
                    
                    # Depth Penalty
                    depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
                    halt_penalty = (depths * (1 - h)).sum(dim=1)
                    
                    progress = global_step / total_steps
                    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
                    lambda_now = lambda_start + (lambda_target - lambda_start) * progress
                    loss_halt = lambda_now * halt_penalty.mean()

                loss = loss_cls + loss_halt
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                xm.optimizer_step(optimizer)
                xm.mark_step()
            
            # --- Logging ---
            loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
            loss_cls_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
            
            if rank == 0:
                elapsed = time.time() - start_time
                xm.master_print(f"[{stage.upper()}] CHUNK {chunk_idx+1} | EPOCH {epoch+1} | Loss: {loss_sum/num_cores:.4f} | Cls: {loss_cls_sum/num_cores:.4f} | Time: {elapsed:.0f}s")

        # --- Checkpoint per chunk ---
        if (chunk_idx + 1) % 5 == 0 and rank == 0:
            checkpoint_path = f"/tmp/controller_{stage}_chunk_{chunk_idx+1}.pt"
            torch.save({
                'chunk': chunk_idx + 1,
                'model_state_dict': model.state_dict(),
            }, checkpoint_path)
            xm.master_print(f"✅ Saved: {checkpoint_path}")

        xm.rendezvous(f"chunk_end_{stage}_{chunk_idx}")
    
    # --- Final Evaluation for this Stage ---
    test_chunk = flags.get("test_chunk", 28)
    evaluate_model(rank, model, test_chunk, 0.8, flags["batch_size"], flags["samples_per_shard"])

# =========================================================================
# ORCHESTRATOR
# =========================================================================
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.torch_xla.device()
    num_cores = xm.xrt_world_size()
    
    # 1. Initialize Model ONCE
    if rank == 0: print("Initializing Controller Model...")
    # NOTE: bias initialized to -2.5 (standard) because we rely on Stage 1 training, not forced bias
    model = Controller(
        L=24,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2,
        halting_bias_init=-2.5 
    ).to(device)

    # Sync initial weights
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.rendezvous("init_sync")

    # =========================================================================
    # STAGE 1: TRAIN TRANSFORMER + CLASSIFIERS (IGNORE HALTING)
    # =========================================================================
    if rank == 0: xm.master_print("\n>>> ENTERING STAGE 1: TRAIN CLASSIFIERS <<<")
    
    # Optimizer for Stage 1: All parameters
    optimizer_cls = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    
    run_stage(rank, flags, model, optimizer_cls, stage="train_cls")
    
    # Save Stage 1 result
    if rank == 0:
        torch.save(model.state_dict(), "/tmp/model_stage1_finished.pt")
        xm.master_print("✅ Stage 1 Complete. Model saved.")
    xm.rendezvous("stage1_done")

    # =========================================================================
    # STAGE 2: TRAIN HALTING HEADS (FREEZE TRANSFORMER/CLS)
    # =========================================================================
    if rank == 0: xm.master_print("\n>>> ENTERING STAGE 2: TRAIN HALTING HEADS <<<")

    # 1. Freeze EVERYTHING
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze HALTING HEADS
    for param in model.halting_heads.parameters():
        param.requires_grad = True

    # 3. Create NEW Optimizer (Only for halting heads)
    optimizer_halt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=flags["lr"], 
        weight_decay=1e-2
    )

    run_stage(rank, flags, model, optimizer_halt, stage="train_halt")
    
    if rank == 0:
        torch.save(model.state_dict(), "/tmp/model_stage2_final.pt")
        xm.master_print("✅ ALL STAGES COMPLETE. Final model saved.")
    xm.rendezvous("job_done")


if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "lr": 1e-4,
        "batch_size": 32, 
        
        "lambda_start": 0.0001,
        "lambda_target": 0.005,
        
        "epochs": 4,
        "samples_per_shard": 19500, 
        
        "test_chunk": 29
    }
    
    print("Starting XLA Two-Stage Training Job...")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')