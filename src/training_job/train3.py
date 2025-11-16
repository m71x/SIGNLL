import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# (1) Import new classes
from torch.utils.data import TensorDataset, DataLoader
from torch_xla.distributed import DistributedSampler

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

def train_loop(rank, flags):
    device = xm.torch_xla.device()
    num_cores = xm.xrt_world_size()

    if rank == 0:
        xm.master_print(f"Detected {num_cores} active cores")
        xm.master_print(f"Starting training with {flags['samples_per_shard']} samples per core")

    # --- Load Data Once (On CPU!) ---
    # Each core loads its own shard
    data = training_data_download(
        core_id=rank,
        filename=flags["chunk_filename"],
        max_entries=flags["samples_per_shard"]
    )
    
    if data is None:
        raise RuntimeError(f"[Core {rank}] Failed to load training data")

    # CRITICAL: Keep data on CPU. DO NOT call .to(device) here.
    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    # Handle layer slicing (still on CPU)
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]
    elif teacher_cls_full.shape[1] != 24:
        raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")

    num_samples = teacher_cls_full.shape[0]

    # --- (2) Create Dataset and DataLoader ---
    
    # Create a standard PyTorch Dataset
    dataset = TensorDataset(teacher_cls_full, teacher_label_full)

    # Create a DistributedSampler to ensure each core gets unique data
    # This also handles shuffling for you.
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_cores,
        rank=rank,
        shuffle=True  # Shuffle data every epoch
    )

    # Create the DataLoader
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=flags["batch_size"],
        drop_last=True,  # CRITICAL: Ensures all batches are the same size
        num_workers=2    # Use a few workers to load data in background
    )

    # CRITICAL: Wait for all cores to finish data setup
    xm.rendezvous("data_prepared")

    if rank == 0:
        xm.master_print(f"All {num_cores} cores have loaded data successfully")
        xm.master_print(f"CLS tokens shape (CPU): {teacher_cls_full.shape}")

    # --- Initialize Model (Same as before) ---
    print(f"[Core {rank}] Initializing model...")
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    # ... (Rest of your weight sync logic, which looks correct) ...
    
    xm.rendezvous("weights_synced")
    if rank == 0:
        xm.master_print("✅ Initial weights synchronized across all cores")

    # ... (Rest of your optimizer setup) ...
    optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none").to(device) # Can move loss fn to device

    # --- Training Configuration ---
    num_epochs = flags["epochs"]
    batch_size = flags["batch_size"]
    
    # (3) Get number of steps from the DataLoader
    # We use len(data_loader) which is XLA-safe
    num_batches_per_epoch = len(data_loader)
    total_steps = num_epochs * num_batches_per_epoch

    if rank == 0:
        xm.master_print(f"Training config: {num_epochs} epochs, {num_batches_per_epoch} batches/epoch")
        xm.master_print(f"Total global samples: (approx) {num_samples * num_cores}")
        xm.master_print(f"Total steps: {total_steps}")
    
    global_step = 0
    start_time = time.time()
    model.train()

    # --- (4) Full Training Loop (Refactored) ---
    for epoch in range(num_epochs):
        if rank == 0:
            xm.master_print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # The sampler handles shuffling, so we just iterate
        for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
            
            # --- (5) Move *this batch* to the device ---
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            
            # --- Loss Calculation (Same as before) ---
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
            
            progress = global_step / total_steps
            lambda_now = lambda_start + (lambda_target - lambda_start) * progress
            loss_halt = lambda_now * halt_penalty.mean()
            
            # Total loss
            loss = loss_cls + loss_halt
            
            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            
            # We still need mark_step()!
            xm.mark_step()
            
            # --- Logging (Same as before) ---
            if global_step % flags["log_interval"] == 0:
                # ... (Your logging logic) ...
                if rank == 0:
                     xm.master_print(f"Epoch: {epoch + 1}/{num_epochs} | Step: {global_step}/{total_steps}")
                     # ...

            global_step += 1
            
        # --- Checkpointing (Same as before) ---
        # ... (Your checkpoint logic) ...
        
        # Ensure all cores finish the epoch before proceeding
        xm.rendezvous(f"epoch_end_{epoch}")

    # ... (Rest of your final checks) ...

# ... (Rest of your _mp_fn and __main__ block) ...