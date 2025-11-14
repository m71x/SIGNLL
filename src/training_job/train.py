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
        
        
            
    # Get batch slice
    start_idx = 1 * batch_size
    end_idx = min(start_idx + batch_size, num_samples)
            
    teacher_cls = teacher_cls_full[start_idx:end_idx]
    teacher_label = teacher_label_full[start_idx:end_idx]
            
    # Forward pass
    halting_logits, class_logits, _ = model(teacher_cls)
    print(halting_logits)
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
    progress = min(1.0, 1 / max(1.0, num_epochs - 1))
    lambda_now = lambda_start + (lambda_target - lambda_start) * progress
    loss_halt = lambda_now * halt_penalty.mean()
        
    # Total loss
    loss = loss_cls + loss_halt
    print(loss)
            
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    xm.optimizer_step(optimizer)
    print("done")
            


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