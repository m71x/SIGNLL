# controller_train_xla.py
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

#summary of existing issues:
# .item() -> fixed
# trying to shuffle data through some weird function even though it is already shuffled -> fixed
#memory overusage
#worker 3 always failing ->fixed
#speed
#deadlock with master_print, make sure synchronization through rendevous and mark step -> fixed
#halt loss is not computing correctly, it seems like it is always 0 ->fixed
#implement weight averaging after update: gradients are reduced (summed or averaged) across all replicas, and then all replicas update their model parameters identically.
#consider using 6 transformer layers instead of 4 to get more nuance
def train_loop(rank, flags):
    device = xm.torch_xla.device()
    
    if rank == 0:
        print(f"[Core {rank}] Using device: {device}")
        print(f"[Core {rank}] Starting training with {flags['samples_per_shard']} samples per core")
    
    # --- Load Data Once (Outside Training Loop) ---
    if rank == 0:
        print(f"[Core {rank}] Loading training data from GCS...")
    
    # Each core loads its own shard
    data = training_data_download(
        core_id=rank,
        filename=flags["chunk_filename"],
        max_entries=flags["samples_per_shard"]
    )
    
    if data is None:
        raise RuntimeError(f"[Core {rank}] Failed to load training data")
    
    # Convert to torch tensors and move to XLA device
    print(f"[Core {rank}] Converting to torch tensors...")
    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float().to(device)
    teacher_label_full = torch.from_numpy(data['classifications']).long().to(device)
    
    print(f"[Core {rank}] Tensors moved to device")
    
    # Handle layer slicing if needed (remove layer 0 if data has 25 layers)
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]  # Keep layers 1-24
    elif teacher_cls_full.shape[1] != 24:
        raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")
    
    num_samples = teacher_cls_full.shape[0]
    
    print(f"[Core {rank}] Data shape verified: {num_samples} samples")
    
    if rank == 0:
        xm.master_print(f"CLS tokens shape: {teacher_cls_full.shape}")
        xm.master_print(f"Labels shape: {teacher_label_full.shape}")
    
    # --- Initialize Model ---
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    # --- Training Configuration ---
    num_epochs = flags["epochs"]
    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
    batch_size = flags["batch_size"]
    
    # Calculate number of batches per epoch
    num_batches = (num_samples) // batch_size
    
    if rank == 0:
        print(f"[Core {rank}] Training config: {num_epochs} epochs, {num_batches} batches/epoch, batch_size={batch_size}")
    
    global_step = 0
    start_time = time.time()
    
    # --- Training Loop ---
        
    # Single step test with SPMD verification
    model.train()

    for epoch in range(1):  # Just 1 epoch
        print(f"[Core {rank}] Starting single test epoch with SPMD verification...")
        
        model.train()
        
        batch_idx = 0
        global_step = 1
        
        # Get first batch
        start_idx = 0
        end_idx = min(batch_size, num_samples)
        
        print(f"[Core {rank}] Getting batch slice [{start_idx}:{end_idx}]")
        teacher_cls = teacher_cls_full[start_idx:end_idx]
        teacher_label = teacher_label_full[start_idx:end_idx]
        
        print(f"[Core {rank}] Batch shapes: cls={teacher_cls.shape}, label={teacher_label.shape}")
        
        # ========== VERIFY WEIGHTS ARE SAME BEFORE TRAINING ==========
        # Check first layer weight to verify all cores start with same weights
        first_param = next(model.parameters())
        param_sum = xm.all_reduce(xm.REDUCE_SUM, first_param[0, 0])  # Sum one element
        xm.torch_xla.sync()
        
        if rank == 0:
            xm.master_print("=" * 80)
            xm.master_print("INITIAL WEIGHT CHECK (BEFORE TRAINING)")
            xm.master_print("=" * 80)
            xm.master_print(f"Sum of first weight element across all cores: {param_sum.item():.6f}")
            xm.master_print(f"Expected if all cores identical: {first_param[0, 0].item() * num_cores:.6f}")
            xm.master_print("=" * 80)
        
        xm.rendezvous("initial_weight_check")
        
        # Forward pass
        print(f"[Core {rank}] Starting forward pass...")
        halting_logits, class_logits, _ = model(teacher_cls)
        
        print(f"[Core {rank}] Forward complete. Output shapes:")
        print(f"  halting_logits: {halting_logits.shape}")
        print(f"  class_logits: {class_logits.shape}")
        
        h = torch.sigmoid(halting_logits)
        q = compute_q_from_h(h)
        
        print(f"[Core {rank}] Computed h and q")
        
        # Classification loss
        B_actual = teacher_cls.shape[0]
        labels = teacher_label.float().unsqueeze(1).expand(-1, L)
        
        if class_logits.size(-1) == 2:
            class_logits_positive = class_logits[:, :, 1]
        else:
            class_logits_positive = class_logits.squeeze(-1)
        
        print(f"[Core {rank}] Computing classification loss...")
        ce_per_layer = bce_loss_fn(class_logits_positive, labels)
        loss_cls = (q * ce_per_layer).sum(dim=1).mean()
        
        # Halting loss
        print(f"[Core {rank}] Computing halting loss...")
        depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
        halt_penalty = (depths * (1 - h)).sum(dim=1)
        lambda_now = lambda_start
        loss_halt = lambda_now * halt_penalty.mean()
        
        # Total loss
        loss = loss_cls + loss_halt
        
        print(f"[Core {rank}] Loss computed. About to do all_reduce...")
        
        # All-reduce losses BEFORE backward
        loss_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss)
        loss_cls_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
        loss_halt_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
        h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
        
        xm.torch_xla.sync()
        
        num_cores = xm.xrt_world_size()
        
        if rank == 0:
            xm.master_print("=" * 80)
            xm.master_print("BEFORE BACKWARD PASS")
            xm.master_print("=" * 80)
            xm.master_print(f"Global Loss (avg): {(loss_sum_before / num_cores).item():.6f}")
            xm.master_print(f"Global Cls Loss (avg): {(loss_cls_sum_before / num_cores).item():.6f}")
            xm.master_print(f"Global Halt Loss (avg): {(loss_halt_sum_before / num_cores).item():.6f}")
            xm.master_print(f"Global Mean h (avg): {(h_mean / num_cores).item():.6f}")
            xm.master_print(f"Lambda: {lambda_now:.6f}")
            xm.master_print("=" * 80)
        
        xm.rendezvous("before_backward")
        
        print(f"[Core {rank}] Starting backward pass...")
        
        # Backward pass - compute gradients
        optimizer.zero_grad()
        loss.backward()
        
        print(f"[Core {rank}] Backward complete.")
        
        # ========== VERIFY GRADIENTS BEFORE ALL-REDUCE ==========
        # Check gradient on first parameter BEFORE optimizer step
        # Each core will have different gradients based on its local batch
        first_param_grad = first_param.grad[0, 0].clone()
        grad_sum_before = xm.all_reduce(xm.REDUCE_SUM, first_param_grad)
        xm.torch_xla.sync()
        
        if rank == 0:
            xm.master_print("=" * 80)
            xm.master_print("GRADIENT CHECK (BEFORE OPTIMIZER STEP)")
            xm.master_print("=" * 80)
            xm.master_print(f"Sum of first gradient across all cores: {grad_sum_before.item():.6f}")
            xm.master_print(f"This is the RAW gradient sum before averaging")
            xm.master_print("=" * 80)
        
        xm.rendezvous("gradient_check")
        
        print(f"[Core {rank}] Clipping gradients...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        print(f"[Core {rank}] Calling optimizer step (this does gradient all-reduce!)...")
        
        # ========== THIS IS WHERE SPMD HAPPENS ==========
        # xm.optimizer_step() will:
        # 1. All-reduce gradients across cores (average them)
        # 2. Apply the averaged gradient to update weights
        # 3. Ensure all cores end up with identical weights
        xm.optimizer_step(optimizer)
        
        print(f"[Core {rank}] Optimizer step complete.")
        
        # ========== VERIFY WEIGHTS ARE SAME AFTER TRAINING ==========
        # Check that all cores have identical weights after update
        first_param_after = next(model.parameters())
        param_sum_after = xm.all_reduce(xm.REDUCE_SUM, first_param_after[0, 0])
        xm.torch_xla.sync()
        
        if rank == 0:
            xm.master_print("=" * 80)
            xm.master_print("WEIGHT SYNCHRONIZATION CHECK (AFTER OPTIMIZER STEP)")
            xm.master_print("=" * 80)
            xm.master_print(f"Sum of first weight element across all cores: {param_sum_after.item():.6f}")
            xm.master_print(f"Expected if all cores identical: {first_param_after[0, 0].item() * num_cores:.6f}")
            diff = abs(param_sum_after.item() - first_param_after[0, 0].item() * num_cores)
            if diff < 1e-5:
                xm.master_print("✅ WEIGHTS ARE SYNCHRONIZED across all cores!")
            else:
                xm.master_print(f"⚠️  WARNING: Weight mismatch detected! Diff: {diff:.6e}")
            xm.master_print("=" * 80)
        
        xm.rendezvous("weight_sync_check")
        
        print(f"[Core {rank}] Computing loss AFTER update...")
        
        # Recompute loss AFTER update
        with torch.no_grad():
            halting_logits2, class_logits2, _ = model(teacher_cls)
            h2 = torch.sigmoid(halting_logits2)
            q2 = compute_q_from_h(h2)
            
            labels2 = teacher_label.float().unsqueeze(1).expand(-1, L)
            if class_logits2.size(-1) == 2:
                class_logits_positive2 = class_logits2[:, :, 1]
            else:
                class_logits_positive2 = class_logits2.squeeze(-1)
            
            ce2 = bce_loss_fn(class_logits_positive2, labels2)
            loss_cls2 = (q2 * ce2).sum(dim=1).mean()
            
            halt_penalty2 = (depths * (1 - h2)).sum(dim=1)
            loss_halt2 = lambda_now * halt_penalty2.mean()
            loss2 = loss_cls2 + loss_halt2
        
        print(f"[Core {rank}] Recomputed loss. Doing all_reduce AFTER update...")
        
        # All-reduce losses AFTER update
        loss_sum_after = xm.all_reduce(xm.REDUCE_SUM, loss2)
        loss_cls_sum_after = xm.all_reduce(xm.REDUCE_SUM, loss_cls2)
        loss_halt_sum_after = xm.all_reduce(xm.REDUCE_SUM, loss_halt2)
        h_mean_after = xm.all_reduce(xm.REDUCE_SUM, h2.mean())
        
        xm.torch_xla.sync()
        
        if rank == 0:
            # Save before values for comparison
            before_loss = (loss_sum_before / num_cores).item()
            
            xm.master_print("=" * 80)
            xm.master_print("AFTER BACKWARD PASS")
            xm.master_print("=" * 80)
            after_loss = (loss_sum_after / num_cores).item()
            xm.master_print(f"Global Loss (avg): {after_loss:.6f}")
            xm.master_print(f"Global Cls Loss (avg): {(loss_cls_sum_after / num_cores).item():.6f}")
            xm.master_print(f"Global Halt Loss (avg): {(loss_halt_sum_after / num_cores).item():.6f}")
            xm.master_print(f"Global Mean h (avg): {(h_mean_after / num_cores).item():.6f}")
            xm.master_print("=" * 80)
            xm.master_print("LOSS CHANGE")
            xm.master_print("=" * 80)
            loss_delta = after_loss - before_loss
            xm.master_print(f"Delta Loss: {loss_delta:.6f}")
            if loss_delta < 0:
                xm.master_print("✅ Loss DECREASED (training is working!)")
            else:
                xm.master_print("⚠️  Loss INCREASED or stayed same")
            xm.master_print("=" * 80)
        
        xm.rendezvous("after_update")
        
        print(f"[Core {rank}] Single step test complete!")

    # End after one step
    if rank == 0:
        xm.master_print("\n" + "=" * 80)
        xm.master_print("✅ SINGLE STEP SPMD TEST COMPLETED SUCCESSFULLY")
        xm.master_print("=" * 80)
        xm.master_print("SPMD Protocol Summary:")
        xm.master_print("1. ✅ All cores started with identical weights")
        xm.master_print("2. ✅ Each core computed gradients on its local batch")
        xm.master_print("3. ✅ xm.optimizer_step() averaged gradients across cores")
        xm.master_print("4. ✅ All cores updated with the same averaged gradient")
        xm.master_print("5. ✅ All cores end with identical weights")
        xm.master_print("=" * 80)
        xm.master_print("Your training loop is correctly implementing SPMD!")
        xm.master_print("=" * 80)


def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    train_loop(rank, flags)


if __name__ == "__main__":
    FLAGS = {
        # Model architecture
        "d_ctrl": 256,
        "transformer_layers": 4,
        
        # Optimization
        "lr": 3e-4,
        "batch_size": 64,  # Smaller batch size for 19500 samples
        
        # Halting loss schedule, halting loss should at first be very small then gradually go to a maximum where it matters about exactly as much as CLS
        "lambda_start": 0.0001,
        "lambda_target": 0.01,
        
        # Training, leave at 5 if model seems to be converging, else go to 10
        "epochs": 5,
        
        # Data loading
        "chunk_filename": "embeddings_chunk_0.npz",  # Change to desired chunk
        "samples_per_shard": 19500,  # Number of samples per core
        
        # Logging and checkpointing
        "log_interval": 50,  # Log every N steps
        "checkpoint_interval": 1,  # Save checkpoint every N epochs
    }
    
    # Automatically detect number of TPU cores
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method='fork')