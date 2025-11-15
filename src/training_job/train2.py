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
#memory overusage ->fixed
#worker 3 always failing ->fixed
#speed ->fixed
#deadlock with master_print, make sure synchronization through rendevous and mark step -> fixed
#halt loss is not computing correctly, it seems like it is always 0 ->fixed
#implement weight averaging after update: gradients are reduced (summed or averaged) across all replicas, and then all replicas update their model parameters identically. ->fixed
#consider using 6 transformer layers instead of 4 to get more nuance
#figure out why code seems to run on a variable amount of cores 25-30 every time you run -> fixed
def train_loop(rank, flags):
    device = xm.torch_xla.device()
    
    # Get actual number of cores (don't hardcode!)
    num_cores = xm.xrt_world_size()
    
    if rank == 0:
        xm.master_print(f"Detected {num_cores} active cores")
        xm.master_print(f"Starting training with {flags['samples_per_shard']} samples per core")
    
    # --- Load Data Once (Outside Training Loop) ---
    print(f"[Core {rank}] Loading training data from GCS...")
    
    # Each core loads its own shard
    data = training_data_download(
        core_id=rank,
        filename=flags["chunk_filename"],
        max_entries=flags["samples_per_shard"]
    )
    
    if data is None:
        raise RuntimeError(f"[Core {rank}] Failed to load training data")
    
    print(f"[Core {rank}] Data loaded from GCS")
    
    # CRITICAL: Wait for all cores to finish loading before proceeding
    xm.rendezvous("data_loaded")
    
    # Convert to torch tensors and move to XLA device
    print(f"[Core {rank}] Converting to torch tensors...")
    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float().to(device)
    teacher_label_full = torch.from_numpy(data['classifications']).long().to(device)
    
    # Handle layer slicing if needed (remove layer 0 if data has 25 layers)
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]  # Keep layers 1-24
    elif teacher_cls_full.shape[1] != 24:
        raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")
    
    num_samples = teacher_cls_full.shape[0]
    
    print(f"[Core {rank}] Data ready: {num_samples} samples")
    
    # Wait for all cores to finish data preparation
    xm.rendezvous("data_prepared")
    
    if rank == 0:
        xm.master_print(f"All {num_cores} cores have loaded data successfully")
        xm.master_print(f"CLS tokens shape: {teacher_cls_full.shape}")
        xm.master_print(f"Labels shape: {teacher_label_full.shape}")
    
    # --- Initialize Model ---
    print(f"[Core {rank}] Initializing model...")
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    print(f"[Core {rank}] Model initialized")
    
    # ========== CRITICAL: SYNCHRONIZE INITIAL WEIGHTS ==========
    if rank == 0:
        xm.master_print("Synchronizing initial weights across all cores...")
    
    # Use DYNAMIC num_cores, not hardcoded!
    for param in model.parameters():
        if rank == 0:
            param.data = param.data * num_cores
        else:
            param.data = param.data * 0
        
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data)
        param.data = param.data / num_cores
    
    xm.mark_step()
    xm.rendezvous("weights_synced")
    
    if rank == 0:
        xm.master_print("✅ Initial weights synchronized across all cores")
    
    # ========== VERIFY INITIAL WEIGHT SYNC ==========
    param_checks = []
    for i, param in enumerate(model.parameters()):
        if i >= 3:
            break
        param_element = param.flatten()[0]
        param_sum = xm.all_reduce(xm.REDUCE_SUM, param_element)
        param_checks.append((param_sum, param_element))
    
    xm.mark_step()
    
    if rank == 0:
        xm.master_print("=" * 80)
        xm.master_print("INITIAL WEIGHT SYNC VERIFICATION")
        xm.master_print("=" * 80)
        for i, (param_sum, param_element) in enumerate(param_checks):
            expected_sum = param_element * num_cores
            diff = param_sum - expected_sum
            xm.master_print(f"Parameter {i}: Difference = {diff}")
        xm.master_print("=" * 80)
    
    xm.rendezvous("verify_weights")
    
    # NOW continue with optimizer initialization
    optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    # --- Training Configuration ---
    num_epochs = flags["epochs"]
    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
    batch_size = flags["batch_size"]
    
    # Calculate number of batches per epoch
    num_batches = num_samples // batch_size
    
    if rank == 0:
        xm.master_print(f"Training config: {num_epochs} epochs, {num_batches} batches/epoch, batch_size={batch_size}")
        xm.master_print(f"Total global samples: {num_samples * num_cores}")
    
    global_step = 0
    start_time = time.time()
    
    # --- Training Loop ---
    model.train()
    
    # Single step test
    for epoch in range(1):
        print(f"[Core {rank}] Starting single test epoch...")
        
        batch_idx = 0
        global_step = 1
        
        # Get first batch
        start_idx = 0
        end_idx = min(batch_size, num_samples)
        
        teacher_cls = teacher_cls_full[start_idx:end_idx]
        teacher_label = teacher_label_full[start_idx:end_idx]
        
        print(f"[Core {rank}] Batch shapes: cls={teacher_cls.shape}, label={teacher_label.shape}")
        
        # Forward pass
        print(f"[Core {rank}] Starting forward pass...")
        halting_logits, class_logits, _ = model(teacher_cls)
        
        h = torch.sigmoid(halting_logits)
        q = compute_q_from_h(h)
        
        # Classification loss
        labels = teacher_label.float().unsqueeze(1).expand(-1, L)
        
        if class_logits.size(-1) == 2:
            class_logits_positive = class_logits[:, :, 1]
        else:
            class_logits_positive = class_logits.squeeze(-1)
        
        ce_per_layer = bce_loss_fn(class_logits_positive, labels)
        loss_cls = (q * ce_per_layer).sum(dim=1).mean()
        
        # Halting loss
        depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
        halt_penalty = (depths * (1 - h)).sum(dim=1)
        lambda_now = lambda_start
        loss_halt = lambda_now * halt_penalty.mean()
        
        # Total loss
        loss = loss_cls + loss_halt
        
        print(f"[Core {rank}] Computing global losses...")
        
        # All-reduce losses BEFORE backward
        loss_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss)
        loss_cls_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
        loss_halt_sum_before = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
        h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
        
        xm.mark_step()
        
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xm.optimizer_step(optimizer)
        
        print(f"[Core {rank}] Optimizer step complete")
        
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
        
        # All-reduce losses AFTER update
        loss_sum_after = xm.all_reduce(xm.REDUCE_SUM, loss2)
        loss_cls_sum_after = xm.all_reduce(xm.REDUCE_SUM, loss_cls2)
        loss_halt_sum_after = xm.all_reduce(xm.REDUCE_SUM, loss_halt2)
        h_mean_after = xm.all_reduce(xm.REDUCE_SUM, h2.mean())
        
        xm.mark_step()
        
        if rank == 0:
            before_loss = (loss_sum_before / num_cores).item()
            after_loss = (loss_sum_after / num_cores).item()
            
            xm.master_print("=" * 80)
            xm.master_print("AFTER BACKWARD PASS")
            xm.master_print("=" * 80)
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
                xm.master_print("✅ Loss DECREASED")
            else:
                xm.master_print("⚠️  Loss INCREASED")
            xm.master_print("=" * 80)
        
        xm.rendezvous("after_update")
        
        print(f"[Core {rank}] Single step test complete!")
    
    # Weight synchronization verification
    if rank == 0:
        xm.master_print("\n" + "=" * 80)
        xm.master_print("FINAL WEIGHT SYNCHRONIZATION CHECK")
        xm.master_print("=" * 80)
    
    param_checks_final = []
    for i, param in enumerate(model.parameters()):
        if i >= 3:
            break
        param_element = param.flatten()[0]
        param_sum = xm.all_reduce(xm.REDUCE_SUM, param_element)
        param_checks_final.append((param_sum, param_element))
    
    xm.mark_step()
    
    if rank == 0:
        for i, (param_sum, param_element) in enumerate(param_checks_final):
            expected_sum = param_element * num_cores
            xm.master_print(f"Parameter {i}:")
            xm.master_print(f"  Sum across cores: {param_sum}")
            xm.master_print(f"  Expected if synced: {expected_sum}")
            xm.master_print(f"  Difference: {param_sum - expected_sum}")
        xm.master_print("=" * 80)
        xm.master_print("✅ TRAINING TEST COMPLETED ON ALL CORES")
        xm.master_print(f"Active cores: {num_cores}/32")
        xm.master_print("=" * 80)
    
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