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
    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float().to(device)
    teacher_label_full = torch.from_numpy(data['classifications']).long().to(device)
    
    # Handle layer slicing if needed (remove layer 0 if data has 25 layers)
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]  # Keep layers 1-24
    elif teacher_cls_full.shape[1] != 24:
        raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")
    
    num_samples = teacher_cls_full.shape[0]
    
    if rank == 0:
        print(f"[Core {rank}] Data loaded: {num_samples} samples")
        print(f"[Core {rank}] CLS tokens shape: {teacher_cls_full.shape}")
        print(f"[Core {rank}] Labels shape: {teacher_label_full.shape}")
    
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
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    if rank == 0:
        print(f"[Core {rank}] Training config: {num_epochs} epochs, {num_batches} batches/epoch, batch_size={batch_size}")
    
    global_step = 0
    start_time = time.time()
    
    # --- Training Loop ---
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_halt_loss = 0.0
        
        # Shuffle data at the start of each epoch
        # IMPORTANT: Create permutation on CPU then move to device (TPU doesn't support int64 RNG)
        perm = torch.randperm(num_samples).to(device)
        teacher_cls_shuffled = teacher_cls_full[perm]
        teacher_label_shuffled = teacher_label_full[perm]
        
        model.train()
        
        for batch_idx in range(num_batches):
            global_step += 1
            
            # Get batch slice
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            teacher_cls = teacher_cls_shuffled[start_idx:end_idx]
            teacher_label = teacher_label_shuffled[start_idx:end_idx]
            
            # Forward pass
            halting_logits, class_logits, _ = model(teacher_cls)
            h = torch.sigmoid(halting_logits)
            q = compute_q_from_h(h)
            
            # Classification loss
            B_actual = teacher_cls.shape[0]
            labels = teacher_label.float().unsqueeze(1).expand(-1, L)
            
            # Handle binary classification (take positive class logit)
            if class_logits.size(-1) == 2:
                class_logits_positive = class_logits[:, :, 1]
            else:
                class_logits_positive = class_logits.squeeze(-1)
            
            ce_per_layer = bce_loss_fn(class_logits_positive, labels)
            loss_cls = (q * ce_per_layer).sum(dim=1).mean()
            
            # Halting loss with linear warmup
            depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
            halt_penalty = (depths * (1 - h)).sum(dim=1)
            progress = min(1.0, epoch / max(1.0, num_epochs - 1))
            lambda_now = lambda_start + (lambda_target - lambda_start) * progress
            loss_halt = lambda_now * halt_penalty.mean()
            
            # Total loss
            loss = loss_cls + loss_halt
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            
            # Accumulate epoch statistics
            epoch_loss += loss.item()
            epoch_cls_loss += loss_cls.item()
            epoch_halt_loss += loss_halt.item()
            
            # Log every N steps
            if global_step % flags["log_interval"] == 0 and rank == 0:
                # Calculate accuracy on this batch
                with torch.no_grad():
                    # Get predicted class (argmax over exit points weighted by q)
                    if class_logits.size(-1) == 2:
                        # Binary classification
                        probs = torch.softmax(class_logits, dim=-1)  # [B, L, 2]
                        weighted_probs = (q.unsqueeze(-1) * probs).sum(dim=1)  # [B, 2]
                        preds = weighted_probs.argmax(dim=-1)
                    else:
                        preds = (class_logits.squeeze(-1) > 0).long()
                    
                    acc = (preds == teacher_label).float().mean().item()
                    mean_depth = (q * depths).sum(dim=1).mean().item()
                
                xm.master_print(
                    f"[Epoch {epoch+1}/{num_epochs}] [Step {global_step}] [Batch {batch_idx+1}/{num_batches}] "
                    f"loss={loss.item():.4f} cls={loss_cls.item():.4f} halt={loss_halt.item():.4f} "
                    f"acc={acc:.4f} mean_h={h.mean().item():.4f} mean_depth={mean_depth:.2f} lambda={lambda_now:.6f}"
                )
        
        # End of epoch logging
        avg_loss = epoch_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_halt_loss = epoch_halt_loss / num_batches
        
        # Global reduce for multi-core statistics
        loss_tensor = torch.tensor([avg_loss, avg_cls_loss, avg_halt_loss], device=device)
        loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss_tensor)
        num_cores = xm.xrt_world_size()
        global_avg_loss, global_avg_cls, global_avg_halt = (loss_sum / num_cores).tolist()
        
        if rank == 0:
            elapsed = time.time() - start_time
            xm.master_print("-" * 80)
            xm.master_print(f"EPOCH {epoch+1}/{num_epochs} COMPLETED")
            xm.master_print(f"  Global Avg Loss: {global_avg_loss:.4f}")
            xm.master_print(f"  Global Avg Cls Loss: {global_avg_cls:.4f}")
            xm.master_print(f"  Global Avg Halt Loss: {global_avg_halt:.4f}")
            xm.master_print(f"  Elapsed Time: {elapsed:.1f}s")
            xm.master_print("-" * 80)
        
        # Save checkpoint every epoch (only on master core)
        if rank == 0 and (epoch + 1) % flags["checkpoint_interval"] == 0:
            checkpoint_path = f"/tmp/controller_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': global_avg_loss,
            }, checkpoint_path)
            xm.master_print(f"✅ Checkpoint saved: {checkpoint_path}")
    
    if rank == 0:
        total_time = time.time() - start_time
        xm.master_print("=" * 80)
        xm.master_print("TRAINING FINISHED")
        xm.master_print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
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