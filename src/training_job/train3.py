import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# (1) Import new classes
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

def train_loop(rank, flags):
    device = xm.torch_xla.device()
    num_cores = xm.xrt_world_size()

    if rank == 0:
        xm.master_print(f"Detected {num_cores} active cores")
        xm.master_print(f"Starting training with {flags['samples_per_shard']} samples per core")

    # --- Load Data Once (On CPU!) ---
    print(f"[Core {rank}] Loading training data from GCS...")
    data = training_data_download(
        core_id=rank,
        filename=flags["chunk_filename"],
        max_entries=flags["samples_per_shard"]
    )
    
    if data is None:
        raise RuntimeError(f"[Core {rank}] Failed to load training data")
    
    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]
    elif teacher_cls_full.shape[1] != 24:
        raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")

    num_samples = teacher_cls_full.shape[0]

    # --- Create Dataset and DataLoader ---
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

    xm.rendezvous("data_prepared")

    if rank == 0:
        xm.master_print(f"All {num_cores} cores have loaded data successfully")
        xm.master_print(f"CLS tokens shape (CPU): {teacher_cls_full.shape}")

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
    
    # --- CRITICAL: SYNCHRONIZE INITIAL WEIGHTS ---
    if rank == 0:
        xm.master_print("Synchronizing initial weights across all cores...")
    
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    
    xm.mark_step()
    xm.rendezvous("weights_synced")
    
    if rank == 0:
        xm.master_print("✅ Initial weights synchronized across all cores")

    # --- VERIFY INITIAL WEIGHT SYNC ---
    param_checks = []
    for i, param in enumerate(model.parameters()):
        if i >= 3:
            break
        # We need to clone and move to CPU for the master print later
        param_element_cpu = param.flatten()[0].cpu() 
        param_checks.append(param_element_cpu)
    
    xm.rendezvous("verify_weights_prep") # Wait for all cores to grab their params
    
    if rank == 0:
        xm.master_print("=" * 80)
        xm.master_print("INITIAL WEIGHT SYNC VERIFICATION (Value on Rank 0)")
        xm.master_print("=" * 80)
        for i, param_val in enumerate(param_checks):
            # This check is simpler: just print the value from Rank 0.
            # The all_reduce above already guarantees they are the same.
            xm.master_print(f"Parameter {i} (Rank 0): Value = {param_val.item():.6f}")
        xm.master_print("=" * 80)
    
    xm.rendezvous("verify_weights_done")
    
    # --- Optimizer Setup ---
    optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none").to(device)

    # --- Training Configuration ---
    num_epochs = flags["epochs"]
    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
    batch_size = flags["batch_size"]
    num_batches_per_epoch = len(data_loader)
    total_steps = num_epochs * num_batches_per_epoch

    if rank == 0:
        xm.master_print(f"Training config: {num_epochs} epochs, {num_batches_per_epoch} batches/epoch")
        xm.master_print(f"Total global samples: (approx) {num_batches_per_epoch * batch_size * num_cores}")
        xm.master_print(f"Total steps: {total_steps}")
    
    global_step = 0
    start_time = time.time()
    model.train()

    # --- Full Training Loop ---
    for epoch in range(num_epochs):
        if rank == 0:
            xm.master_print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        sampler.set_epoch(epoch)

        # --- FIX: Initialize as tensors on device ---
        epoch_total_loss = torch.tensor(0.0, device=device)
        epoch_cls_loss = torch.tensor(0.0, device=device)
        epoch_halt_loss = torch.tensor(0.0, device=device)
        epoch_h_mean = torch.tensor(0.0, device=device)

        for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
            
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            
            # --- Loss Calculation ---
            halting_logits, class_logits, _ = model(teacher_cls)
            
            h = torch.sigmoid(halting_logits)
            q = compute_q_from_h(h)
            
            labels = teacher_label.float().unsqueeze(1).expand(-1, L)
            
            if class_logits.size(-1) == 2:
                class_logits_positive = class_logits[:, :, 1]
            else:
                class_logits_positive = class_logits.squeeze(-1)
            
            ce_per_layer = bce_loss_fn(class_logits_positive, labels)
            loss_cls = (q * ce_per_layer).sum(dim=1).mean()
            
            depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
            halt_penalty = (depths * (1 - h)).sum(dim=1)
            
            progress = global_step / total_steps
            lambda_now = lambda_start + (lambda_target - lambda_start) * progress
            loss_halt = lambda_now * halt_penalty.mean()
            
            loss = loss_cls + loss_halt
            
            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            
            # We still need mark_step()!
            xm.mark_step()
            
            # --- FIX: Accumulate as tensors (NO .item()) ---
            # We use .detach() to prevent gradient history from accumulating
            epoch_total_loss += loss.detach()
            epoch_cls_loss += loss_cls.detach()
            epoch_halt_loss += loss_halt.detach()
            epoch_h_mean += h.mean().detach()

            # --- Logging (Per-Step) ---
            if global_step % flags["log_interval"] == 0:
                
                # All-reduce losses for logging
                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                loss_cls_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
                loss_halt_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
                h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
                
                # Log on master core after all-reduce
                if rank == 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    xm.master_print("-" * 50)
                    xm.master_print(f"Epoch: {epoch + 1}/{num_epochs} | Step: {global_step}/{total_steps} ({global_step * 100 / total_steps:.1f}%)")
                    xm.master_print(f"Avg Total Loss: {(loss_sum / num_cores).item():.6f}")
                    xm.master_print(f"Avg Cls Loss: {(loss_cls_sum / num_cores).item():.6f}")
                    xm.master_print(f"Avg Halt Loss: {(loss_halt_sum / num_cores).item():.6f}")
                    xm.master_print(f"Avg Mean h: {(h_mean / num_cores).item():.6f}")
                    xm.master_print(f"Lambda: {lambda_now:.6f}")
                    xm.master_print(f"Time elapsed: {elapsed_time:.1f}s")
                    xm.master_print("-" * 50)

            global_step += 1
        
        # --- End of Epoch Logging and Checkpoint ---

        # All-reduce the accumulated epoch totals
        total_loss_all_reduce = xm.all_reduce(xm.REDUCE_SUM, epoch_total_loss)
        cls_loss_all_reduce = xm.all_reduce(xm.REDUCE_SUM, epoch_cls_loss)
        halt_loss_all_reduce = xm.all_reduce(xm.REDUCE_SUM, epoch_halt_loss)
        h_mean_all_reduce = xm.all_reduce(xm.REDUCE_SUM, epoch_h_mean)
        
        if rank == 0:
            # We divide by num_batches_per_epoch here because the tensors 
            # already represent the sum of losses *on each core*.
            avg_epoch_loss = (total_loss_all_reduce / (num_cores * num_batches_per_epoch)).item()
            avg_epoch_cls_loss = (cls_loss_all_reduce / (num_cores * num_batches_per_epoch)).item()
            avg_epoch_halt_loss = (halt_loss_all_reduce / (num_cores * num_batches_per_epoch)).item()
            avg_epoch_h = (h_mean_all_reduce / (num_cores * num_batches_per_epoch)).item()

            xm.master_print(f"\n======== EPOCH {epoch + 1} SUMMARY ========")
            xm.master_print(f"   AVG Epoch Total Loss: {avg_epoch_loss:.6f}")
            xm.master_print(f"   AVG Epoch CLS Loss:   {avg_epoch_cls_loss:.6f}")
            xm.master_print(f"   AVG Epoch Halt Loss:  {avg_epoch_halt_loss:.6f}")
            xm.master_print(f"   AVG Epoch Mean h:     {avg_epoch_h:.6f}")
            xm.master_print(f"========================================\n")


        if (epoch + 1) % flags["checkpoint_interval"] == 0:
            if rank == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
                # xm.save(model.state_dict(), checkpoint_path) # Example save command
                xm.master_print(f"Checkpoint saved for Epoch {epoch + 1} at {checkpoint_path} (placeholder)")
        
        xm.rendezvous(f"epoch_end_{epoch}")

    # --- FINAL WEIGHT CHECK ---
    total_time = time.time() - start_time
    if rank == 0:
        xm.master_print("\n" + "=" * 80)
        xm.master_print("TRAINING FINISHED")
        xm.master_print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        xm.master_print(f"Total steps: {global_step}")
        xm.master_print("=" * 80)
    
    # Final check is good for debugging
    param_checks_final_cpu = []
    for i, param in enumerate(model.parameters()):
        if i >= 3:
            break
        param_element_cpu = param.flatten()[0].cpu()
        param_checks_final_cpu.append(param_element_cpu)
    
    xm.rendezvous("final_check_prep")

    if rank == 0:
        xm.master_print("FINAL WEIGHT VERIFICATION (Value on Rank 0)")
        for i, param_val in enumerate(param_checks_final_cpu):
            xm.master_print(f"Parameter {i} (Rank 0): Value = {param_val.item():.6f}")
        xm.master_print("=" * 80)
        xm.master_print(f"✅ FINAL TRAINING COMPLETED ON ALL CORES. Active cores: {num_cores}")
        xm.master_print("=" * 80)
    
    xm.rendezvous("final_check_done")
    
def _mp_fn(rank, flags):
    try:
        torch.set_default_tensor_type('torch.FloatTensor')
        train_loop(rank, flags)
    except Exception as e:
        print(f"[Core {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # If one core raises an exception, we need to ensure the TPU job stops
        # Calling xm.rendezvous here might hang if other cores are stuck,
        # so simply re-raising the exception is the right move for XLA.
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