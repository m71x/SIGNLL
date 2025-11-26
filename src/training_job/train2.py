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
# Be wary of using torch.save()

# =========================================================================
# NEW FUNCTION: Evaluation / Testing
# =========================================================================
def evaluate_model(rank, model, chunk_idx, threshold, batch_size, samples_per_shard):
    """
    Tests the model on a specific chunk using an early-exit strategy.
    Runs ONLY on Device 0.
    """
    # STRICT GUARD: Only run on Rank 0
    if rank != 0:
        return

    device = xm.xla_device()
    
    xm.master_print(f"\n{'*'*80}")
    xm.master_print(f"*** STARTING EVALUATION ON CHUNK {chunk_idx} (Device 0 Only) ***")
    xm.master_print(f"*** Early Exit Threshold: {threshold} ***")
    xm.master_print(f"{'*'*80}")

    model.eval()
    
    # --- Load Data (Reusing logic from train_loop) ---
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
    
    # Use SequentialSampler since we are only running on one device and want to check all loaded data
    sampler = SequentialSampler(dataset)
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2
    )
    
    # Initialize accumulators as CPU INTEGERS/FLOATS, not XLA Tensors
    total_samples = 0
    total_correct = 0
    
    # Track which layer triggered the exit (Keep on CPU)
    layer_exit_counts_cpu = torch.zeros(24, dtype=torch.float32)

    with torch.no_grad():
        for i, (teacher_cls, teacher_label) in enumerate(data_loader):
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            
            # Forward pass
            halting_logits, class_logits, _ = model(teacher_cls)
            
            # --- Early Exit Logic ---
            h_probs = torch.sigmoid(halting_logits)
            threshold_mask = (h_probs > threshold)
            
            # Find the index of the FIRST layer that exceeds threshold
            exit_indices = torch.argmax(threshold_mask.long(), dim=1)
            
            # Handle samples that NEVER crossed the threshold (force to last layer)
            row_has_exit = threshold_mask.any(dim=1)
            exit_indices[~row_has_exit] = 23
            
            # --- Classification ---
            batch_indices = torch.arange(class_logits.size(0), device=device)
            selected_logits = class_logits[batch_indices, exit_indices]
            predictions = torch.argmax(selected_logits, dim=-1)
            
            # --- Metrics Calculation (FIX FOR HANG) ---
            correct_tensor = (predictions == teacher_label).sum()
            
            # CRITICAL: Move results to CPU immediately to break the XLA graph.
            # If we keep adding tensors (total_correct += correct_tensor), the graph grows infinitely.
            total_correct += correct_tensor.item() 
            total_samples += teacher_label.size(0)
            
            # Move exit indices to CPU for statistics
            exit_indices_cpu = exit_indices.cpu()
            unique_exits, counts = torch.unique(exit_indices_cpu, return_counts=True)
            layer_exit_counts_cpu.index_add_(0, unique_exits, counts.float())
            
            # Trigger execution of this batch
            xm.mark_step()
            
            if i % 100 == 0:
                print(f"[Eval] Processed batch {i}...")
            
    # --- Calculate Results (Local only) ---
    accuracy = (total_correct / total_samples) * 100.0
    
    # Calculate average exit layer using CPU tensors
    layers = torch.arange(24, dtype=torch.float32)
    avg_exit_layer = (layer_exit_counts_cpu * layers).sum() / total_samples
    
    xm.master_print(f"RESULTS FOR CHUNK {chunk_idx} (Threshold: {threshold}):")
    xm.master_print(f"  Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")
    xm.master_print(f"  Average Exit Layer: {avg_exit_layer:.2f} (0-23)")
    xm.master_print(f"{'*'*80}\n")

    model.train() # Reset to train mode

def train_loop(rank, flags):
    """
    The main training function executed independently on each TPU core.
    This function initializes the model once and then iterates through all 29 data chunks (0-28) sequentially.
    """
    device = xm.torch_xla.device()
    num_cores = xm.xrt_world_size()
    stage = flags["training_stage"] # "train_cls" or "train_halt"

    if rank == 0:
        xm.master_print(f"Detected {num_cores} active cores")
        xm.master_print(f"Starting Training in STAGE: {stage.upper()}")
        xm.master_print(f"Starting single-model training across 29 sequential data chunks.")
        
    # =========================================================================
    # 1. MODEL INITIALIZATION (RUNS ONCE PER CORE, OUTSIDE CHUNK LOOP)
    # =========================================================================
    print(f"[Core {rank}] Initializing model...")
    L = 24
    
    # Reset bias to standard (-2.5) because we handle flow via stages now
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2,
        halting_bias_init=-2.5
    ).to(device)
    
    # --- STAGE 2: FREEZING LOGIC ---
    if stage == "train_halt":
        if rank == 0:
            xm.master_print("STAGE 2 DETECTED: Freezing Transformer and Classifiers. Training ONLY Halting Heads.")
            
        # Freeze everything first
        for name, param in model.named_parameters():
            param.requires_grad = False
            
        # Unfreeze ONLY halting heads
        for name, param in model.halting_heads.named_parameters():
            param.requires_grad = True
            if rank == 0:
                print(f"  -> Unfrozen: {name}")
                
        # Optional: Unfreeze projection/embedding if you want them adaptable
        # usually better to freeze them too so the state is stable.
    else:
        if rank == 0:
            xm.master_print("STAGE 1 DETECTED: Training ALL parameters (Transformer + Classifiers).")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=flags["lr"], weight_decay=1e-2)
    
    # CRITICAL: SYNCHRONIZE INITIAL WEIGHTS ACROSS ALL CORES
    if rank == 0:
        xm.master_print("Synchronizing initial weights across all cores...")
    
    # Clean synchronization implementation
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    
    xm.mark_step()
    xm.rendezvous("weights_synced")
    
    # --- STAGE 2: LOAD WEIGHTS (Simulated) ---
    # In a real run, you would load the checkpoint from Stage 1 here.
    # For now, we assume you run Stage 1, save, change flag, and restart script loading checkpoint.
    # if stage == "train_halt":
    #    checkpoint = torch.load("/tmp/stage1_checkpoint.pt")
    #    model.load_state_dict(checkpoint['model_state_dict'])
    
    if rank == 0:
        xm.master_print("✅ Initial weights synchronized across all cores. Starting chunk loop.")

    # Variables to hold diagnostic data for rank 0 (re-used for every chunk)
    sample_logits_pos_cpu = None
    sample_label_pos_cpu = None
    sample_logits_neg_cpu = None
    sample_label_neg_cpu = None
    
    # Calculate total steps across ALL chunks for lambda schedule
    total_samples = flags["samples_per_shard"] #* num_cores
    num_batches_per_chunk = total_samples // flags["batch_size"]
    total_steps = flags["epochs"] * 29 * num_batches_per_chunk
    
    global_step = 0
    start_time = time.time()
    model.train()

    # =========================================================================
    # 2. OUTER LOOP: ITERATE OVER DATA CHUNKS (0 to 28)
    # =========================================================================
    for chunk_idx in range(29): 
        current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
        
        if rank == 0:
            xm.master_print(f"\n{'#'*90}")
            xm.master_print(f"### CHUNK {chunk_idx + 1}/29: Loading data from {current_chunk_filename} ###")
            xm.master_print(f"{'#'*90}")

        # --- Load Data For Current Chunk (On CPU!) ---
        data = training_data_download(
            core_id=rank,
            filename=current_chunk_filename,
            max_entries=flags["samples_per_shard"]
        )
        
        if data is None:
            raise RuntimeError(f"[Core {rank}] Failed to load training data for chunk {chunk_idx}")

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        
        # Handle layer slicing
        if teacher_cls_full.shape[1] == 25:
            teacher_cls_full = teacher_cls_full[:, 1:25, :]
        elif teacher_cls_full.shape[1] != 24:
            raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")

        # === Calculate and Apply Class Weighting (Per-Chunk) ===
        neg_samples_count = (teacher_label_full == 0).sum().item()
        pos_samples_count = (teacher_label_full == 1).sum().item()
        
        pos_weight_value = neg_samples_count / pos_samples_count
        pos_weight_tensor = torch.tensor([pos_weight_value]).float()
        pos_weight_device = pos_weight_tensor.to(device)

        if rank == 0:
            xm.master_print(f"[Core {rank}] Chunk {chunk_idx+1} Positive Weight: {pos_weight_value:.4f}")

        # Instantiate BCE loss with the calculated weight (pos_weight)
        bce_loss_fn = nn.BCEWithLogitsLoss(
            reduction="none", 
            pos_weight=pos_weight_device
        ).to(device)

        # --- Create Dataset and DataLoader (Per-Chunk) ---
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
        
        # CRITICAL: Wait for all cores to finish data setup for this chunk
        xm.rendezvous(f"data_prepared_chunk_{chunk_idx}")

        # =========================================================================
        # 3. MIDDLE LOOP: ITERATE OVER EPOCHS (1 to 10)
        # =========================================================================
        for epoch in range(flags["epochs"]):
            if rank == 0:
                xm.master_print(f"\n{'='*80}")
                xm.master_print(f"CHUNK {chunk_idx + 1}/29 | EPOCH {epoch + 1}/{flags['epochs']}")
                xm.master_print(f"{'='*80}")
            
            for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                global_step += 29
                # --- Move *this batch* to the device ---
                teacher_cls = teacher_cls.to(device)
                teacher_label = teacher_label.to(device)
                
                # --- Forward Pass ---
                halting_logits, class_logits, _ = model(teacher_cls)
                
                # === Capture diagnostic data ===
                if rank == 0 and batch_idx == 0:
                    positive_indices = (teacher_label == 1).nonzero(as_tuple=True)[0]
                    negative_indices = (teacher_label == 0).nonzero(as_tuple=True)[0]

                    if positive_indices.numel() > 0:
                        pos_idx = positive_indices[0]
                        sample_logits_pos_cpu = class_logits[pos_idx].detach().cpu() 
                        sample_label_pos_cpu = teacher_label[pos_idx].detach().cpu()
                    else:
                        sample_logits_pos_cpu = None
                        sample_label_pos_cpu = None

                    if negative_indices.numel() > 0:
                        neg_idx = negative_indices[0]
                        sample_logits_neg_cpu = class_logits[neg_idx].detach().cpu()
                        sample_label_neg_cpu = teacher_label[neg_idx].detach().cpu()
                    else:
                        sample_logits_neg_cpu = None
                        sample_label_neg_cpu = None
                
                # --- Loss Calculation ---
                labels = teacher_label.float().unsqueeze(1).expand(-1, L)
                if class_logits.size(-1) == 2:
                    class_logits_positive = class_logits[:, :, 1]
                else:
                    class_logits_positive = class_logits.squeeze(-1)
                
                ce_per_layer = bce_loss_fn(class_logits_positive, labels)
                
                # --- STAGE-SPECIFIC LOSS ---
                if stage == "train_cls":
                    # STAGE 1: Train ALL classifiers. Ignore halting.
                    loss_cls = ce_per_layer.mean()
                    loss_halt = torch.tensor(0.0, device=device)
                    # We compute h just for logs, but don't use it for loss
                    h = torch.sigmoid(halting_logits) 
                    
                elif stage == "train_halt":
                    # STAGE 2: Train Halting Heads to match Classifier performance
                    h = torch.sigmoid(halting_logits)
                    q = compute_q_from_h(h)
                    
                    # We multiply q * error.
                    # Since classifiers are frozen, we are optimizing q to put mass where error is LOW.
                    loss_cls = (q * ce_per_layer).sum(dim=1).mean()
                    
                    # Add Depth Penalty
                    depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
                    halt_penalty = (depths * (1 - h)).sum(dim=1)
                    
                    progress = global_step / total_steps
                    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
                    lambda_now = lambda_start + (lambda_target - lambda_start) * progress
                    loss_halt = lambda_now * halt_penalty.mean()

                loss = loss_cls + loss_halt
                
                # --- Backward Pass and Optimization ---
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                xm.optimizer_step(optimizer)
                
                xm.mark_step()
            
            # --- End of Epoch Diagnostics ---
            loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
            loss_cls_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
            loss_halt_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
            h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
            
            xm.mark_step()
            
            if rank == 0:
                def format_diagnostic_output(logits_cpu, label_cpu, sample_type):
                    if logits_cpu is None or label_cpu is None:
                        return f"  No {sample_type} sample found in the first batch."
                    sample_probs = torch.softmax(logits_cpu, dim=-1)
                    predicted_classes = torch.argmax(sample_probs, dim=-1).tolist()
                    max_confidences = sample_probs.max(dim=-1).values
                    predicted_probs = [f"{p:.4f}" for p in max_confidences.tolist()]
                    true_label_value = label_cpu.item()
                    output = []
                    output.append(f"  --- {sample_type} Sample (True Label: {true_label_value}) ---")
                    output.append(f"  Predicted Class per Layer (0-23): {predicted_classes}")
                    output.append(f"  Max Confidence per Layer: {predicted_probs}")
                    return "\n".join(output)
                
                elapsed = time.time() - start_time
                xm.master_print("-" * 80)
                xm.master_print(f"EPOCH {epoch+1}/{flags['epochs']} COMPLETED for CHUNK {chunk_idx + 1}")
                xm.master_print(f"  Stage: {stage}")
                xm.master_print("FINAL METRICS:")
                xm.master_print(f"  Avg Total Loss: {loss_sum / num_cores}")
                xm.master_print(f"  Avg Cls Loss: {loss_cls_sum / num_cores}")
                if stage == "train_halt":
                    xm.master_print(f"  Avg Halt Loss: {loss_halt_sum / num_cores}")
                    xm.master_print(f"  Lambda: {lambda_now:.6f}")
                xm.master_print(f"  Avg Mean h: {h_mean / num_cores}")
                
                if sample_logits_pos_cpu is not None:
                    xm.master_print("\nDUAL SAMPLE CLASSIFICATION DIAGNOSTIC (from first batch):")
                    pos_output = format_diagnostic_output(sample_logits_pos_cpu, sample_label_pos_cpu, "Positive (Label 1)")
                    xm.master_print(pos_output)
                    neg_output = format_diagnostic_output(sample_logits_neg_cpu, sample_label_neg_cpu, "Negative (Label 0)")
                    xm.master_print(neg_output)
                
                xm.master_print("-" * 80)
            
            xm.rendezvous(f"epoch_end_chunk_{chunk_idx}_epoch_{epoch}")
        
        # --- Checkpointing ---
        if (chunk_idx + 1) % 5 == 0 and rank == 0:
            checkpoint_path = f"/tmp/controller_{stage}_chunk_{chunk_idx+1}_final.pt"
            torch.save({
                'chunk': chunk_idx + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            xm.master_print(f"✅ CHUNK Checkpoint saved: {checkpoint_path}")
        
        xm.rendezvous(f"chunk_end_{chunk_idx}")

    # --- FINAL SUMMARY ---
    total_time = time.time() - start_time
    if rank == 0:
        xm.master_print(f"✅ STAGE {stage} COMPLETED.")
        final_checkpoint_path = f"/tmp/controller_{stage}_final_29_chunks.pt"
        torch.save({
            'chunk': 29,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_checkpoint_path)
        xm.master_print(f"✅ FINAL Checkpoint saved: {final_checkpoint_path}")
        
    # =========================================================================
    # CALL EVALUATION FUNCTION
    # =========================================================================
    evaluate_model(
        rank=rank,
        model=model,
        chunk_idx=flags.get("test_chunk", 28),
        threshold=0.7,
        batch_size=flags["batch_size"],
        samples_per_shard=flags["samples_per_shard"]
    )
    
    xm.rendezvous("final_check")

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
        # --- TRAINING STAGE CONTROL ---
        # OPTIONS: "train_cls" (First) OR "train_halt" (Second)
        "training_stage": "train_cls", 
        
        "d_ctrl": 512,
        "transformer_layers": 4,
        "lr": 1e-4,
        "batch_size": 32, 
        
        "lambda_start": 0.0001,
        "lambda_target": 0.005, # Higher target is safe in Stage 2
        
        "epochs": 4,
        "samples_per_shard": 19500, 
        
        "test_chunk": 28,     
        "test_threshold": 0.8 
    }
    
    print(f"Starting single XLA spawn job. STAGE: {BASE_FLAGS['training_stage']}")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')
    print("XLA spawn job completed.")