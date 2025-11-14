# controller_train_xla.py
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

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
    
        
        # Shuffle data at the start of each epoch
        # IMPORTANT: Create permutation on CPU then move to device (TPU doesn't support int64 RNG)
        #perm = torch.randperm(num_samples).to(device)
        #teacher_cls_shuffled = teacher_cls_full[perm]
        #teacher_label_shuffled = teacher_label_full[perm]
        
    model.train()
        
        
            
    for epoch in range(1):  # Just 1 epoch
        print(f"[Core {rank}] Starting single test epoch...")
    
        model.train()
        
        # Just do ONE batch
        batch_idx = 0
        global_step = 1
        
        # Get first batch
        start_idx = 0
        end_idx = min(batch_size, num_samples)
        
        print(f"[Core {rank}] Getting batch slice [{start_idx}:{end_idx}]")
        teacher_cls = teacher_cls_full[start_idx:end_idx]
        teacher_label = teacher_label_full[start_idx:end_idx]
        
        print(f"[Core {rank}] Batch shapes: cls={teacher_cls.shape}, label={teacher_label.shape}")
        
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
        
        # Handle binary classification (take positive class logit)
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
        progress = 0.0  # First epoch
        lambda_now = lambda_start
        loss_halt = lambda_now * halt_penalty.mean()
        
        # Total loss
        loss = loss_cls + loss_halt
        
        print(f"[Core {rank}] Loss computed. About to do all_reduce...")
        
        # All-reduce losses BEFORE backward (to see initial state)
        loss_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss)
        loss_cls_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
        loss_halt_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
        h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
        
        num_cores = xm.xrt_world_size()
        
        if rank == 0:
            xm.master_print("=" * 80)
            xm.master_print("BEFORE BACKWARD PASS")
            xm.master_print("=" * 80)
            xm.master_print(f"Global Loss (avg): {loss_sum_before / num_cores}")
            xm.master_print(f"Global Cls Loss (avg): {loss_cls_sum_before / num_cores}")
            xm.master_print(f"Global Halt Loss (avg): {loss_halt_sum_before / num_cores}")
            xm.master_print(f"Global Mean h (avg): {h_mean / num_cores}")
            xm.master_print(f"Lambda: {lambda_now}")
            xm.master_print("=" * 80)
        
        print(f"[Core {rank}] Starting backward pass...")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        print(f"[Core {rank}] Backward complete. Clipping gradients...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        print(f"[Core {rank}] Calling optimizer step...")
        xm.optimizer_step(optimizer)
        
        print(f"[Core {rank}] Optimizer step complete. Computing loss AFTER update...")
        
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
        
        if rank == 0:
            xm.master_print("=" * 80)
            xm.master_print("AFTER BACKWARD PASS")
            xm.master_print("=" * 80)
            xm.master_print(f"Global Loss (avg): {loss_sum_after / num_cores}")
            xm.master_print(f"Global Cls Loss (avg): {loss_cls_sum_after / num_cores}")
            xm.master_print(f"Global Halt Loss (avg): {loss_halt_sum_after / num_cores}")
            xm.master_print(f"Global Mean h (avg): {h_mean_after / num_cores}")
            xm.master_print("=" * 80)
            xm.master_print("LOSS CHANGE")
            xm.master_print("=" * 80)
            loss_delta = (loss_sum_after - loss_sum_before) / num_cores
            xm.master_print(f"Delta Loss: {loss_delta}")
            # Note: Can't do if/else on XLA tensors without .item(), just print the delta
            xm.master_print("(Check if negative = loss decreased)")
            xm.master_print("=" * 80)
        
        print(f"[Core {rank}] Single step test complete!")

    # End after one step
    if rank == 0:
        xm.master_print("\n✅ SINGLE STEP TEST COMPLETED SUCCESSFULLY")
        xm.master_print("If you see this message, the training loop is working correctly!")
            


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
        
        # Halting loss schedule
        "lambda_start": 0.0,
        "lambda_target": 0.01,
        
        # Training
        "epochs": 10,
        
        # Data loading
        "chunk_filename": "embeddings_chunk_0.npz",  # Change to desired chunk
        "samples_per_shard": 19500,  # Number of samples per core
        
        # Logging and checkpointing
        "log_interval": 50,  # Log every N steps
        "checkpoint_interval": 1,  # Save checkpoint every N epochs
    }
    
    # Automatically detect number of TPU cores
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method='fork')