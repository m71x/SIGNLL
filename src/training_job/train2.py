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
#NOTE:
#DO NOT USE .item(), IT WILL FORCE XLA TO RECOMPILE AT EVERY BATCH ITERATION
#be wary of using torch.save()
#TODO
#write code to run on all chunks from 0-28 for each shard
#double check architecture of model and loss is actually what you want it to do
#fix the loss function to prioritize CLS loss more, CLS loss seems to be increasing per epoch because halt loss matters too much
#modify script to also show an example of predicted classification vs label for each of the 24 CLS/halting heads per epoch
#consider if it is necessary to do data sharding so that the number of positive and negative samples are approximately equal, right now it is more of a 3-1 ratio
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

    # === NEW: Print Sample Distribution for the current shard ===
    # Calculate the number of samples with label 0
    # Use .item() to extract the Python integer from the single-element tensor for printing
    neg_samples_count = (teacher_label_full == 0).sum().item() 
    
    # Use standard print() here to show local core data distribution
    print(f"[Core {rank}] Data Shard Check: {neg_samples_count} samples have Label 0 (Negative).")
    # ==========================================================
    
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
        shuffle=True # Shuffle data every epoch
    )

    # Create the DataLoader
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=flags["batch_size"],
        drop_last=True, # CRITICAL: Ensures all batches are the same size
        num_workers=2  # Use a few workers to load data in background
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
    
    # ========== CRITICAL: SYNCHRONIZE INITIAL WEIGHTS ==========
    if rank == 0:
        xm.master_print("Synchronizing initial weights across all cores...")
    
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

    # ... (Rest of your optimizer setup) ...
    optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none").to(device) # Can move loss fn to device

    # --- Training Configuration ---
    num_epochs = flags["epochs"]
    batch_size = flags["batch_size"]
    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
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

    # Variables to hold diagnostic data for rank 0
    sample_logits_cpu = None
    sample_label_cpu = None
    
    # NEW: Variables to hold one positive and one negative sample diagnostic data
    sample_logits_pos_cpu = None
    sample_label_pos_cpu = None
    sample_logits_neg_cpu = None
    sample_label_neg_cpu = None
    # ==========================================================

    # --- (4) Full Training Loop (Refactored) ---
    for epoch in range(num_epochs):
        if rank == 0:
            xm.master_print(f"\n{'='*80}")
            xm.master_print(f"EPOCH {epoch + 1}/{num_epochs}")
            xm.master_print(f"{'='*80}")
        
        # The sampler handles shuffling, so we just iterate
        for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
            
            # --- (5) Move *this batch* to the device ---
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            
            # --- Loss Calculation (Same as before) ---
            halting_logits, class_logits, _ = model(teacher_cls)
            
            # === Capture diagnostic data on the first batch of rank 0 (MODIFIED) ===
            if rank == 0 and batch_idx == 0:
                # Keep existing variables assigned to comply with 'don't delete'
                sample_logits_cpu = class_logits[0].detach().cpu()
                sample_label_cpu = teacher_label[0].detach().cpu()
                
                # NEW: Find and capture one positive and one negative sample
                positive_indices = (teacher_label == 1).nonzero(as_tuple=True)[0]
                negative_indices = (teacher_label == 0).nonzero(as_tuple=True)[0]

                # Capture Positive Sample (Label 1)
                if positive_indices.numel() > 0:
                    pos_idx = positive_indices[0]
                    # Logits are (L, 2). Extract the specific sample and move to CPU.
                    sample_logits_pos_cpu = class_logits[pos_idx].detach().cpu() 
                    sample_label_pos_cpu = teacher_label[pos_idx].detach().cpu()
                else:
                    sample_logits_pos_cpu = None
                    sample_label_pos_cpu = None

                # Capture Negative Sample (Label 0)
                if negative_indices.numel() > 0:
                    neg_idx = negative_indices[0]
                    # Logits are (L, 2). Extract the specific sample and move to CPU.
                    sample_logits_neg_cpu = class_logits[neg_idx].detach().cpu()
                    sample_label_neg_cpu = teacher_label[neg_idx].detach().cpu()
                else:
                    sample_logits_neg_cpu = None
                    sample_label_neg_cpu = None
            
            # --- Loss Calculation (Continuing) ---
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
            #deleted lambda increase for now
            lambda_now = lambda_start
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
            
            global_step += 1
        
        # --- End of Epoch Diagnostics ---
        # All-reduce losses for epoch summary
        loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
        loss_cls_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
        loss_halt_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
        h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
        
        # Weight sync check (keeping for debugging consistency)
        param_checks = []
        for i, param in enumerate(model.parameters()):
            if i >= 3:
                break
            param_element = param.flatten()[0]
            param_sum = xm.all_reduce(xm.REDUCE_SUM, param_element)
            param_checks.append((param_sum, param_element))
        
        xm.mark_step()
        
        if rank == 0:
            
            # NEW HELPER FUNCTION DEFINITION
            def format_diagnostic_output(logits_cpu, label_cpu, sample_type):
                """Helper function to format the prediction vs label output for a single sample."""
                if logits_cpu is None or label_cpu is None:
                    return f"  No {sample_type} sample found in the first batch of this core."

                # Logits shape is (24, 2)
                sample_probs = torch.softmax(logits_cpu, dim=-1)
                
                # Get the predicted class index (0 or 1) for each of the 24 layers
                predicted_classes = torch.argmax(sample_probs, dim=-1).tolist()
                
                # Get the maximum confidence for all layers as a tensor (24,)
                max_confidences = sample_probs.max(dim=-1).values
                max_confidences_list = max_confidences.tolist()
                predicted_probs = [f"{p:.4f}" for p in max_confidences_list]
                
                true_label_value = label_cpu.item()
                
                output = []
                output.append(f" --- {sample_type} Sample (True Label: {true_label_value}) ---")
                output.append(f" Predicted Class per Layer (0-23): {predicted_classes}")
                output.append(f" Max Confidence per Layer: {predicted_probs}")
                return "\n".join(output)
            # END NEW HELPER FUNCTION
            
            elapsed = time.time() - start_time
            xm.master_print("-" * 80)
            xm.master_print(f"EPOCH {epoch+1}/{num_epochs} COMPLETED")
            xm.master_print(f"  Total Elapsed Time: {elapsed:.1f}s")
            xm.master_print("")
            xm.master_print("FINAL METRICS:")
            xm.master_print(f"  Step: {global_step}/{total_steps} ({global_step * 100 / total_steps:.1f}%)")
            xm.master_print(f"  Avg Total Loss: {loss_sum / num_cores}")
            xm.master_print(f"  Avg Cls Loss: {loss_cls_sum / num_cores}")
            xm.master_print(f"  Avg Halt Loss: {loss_halt_sum / num_cores}")
            xm.master_print(f"  Avg Mean h: {h_mean / num_cores}")
            xm.master_print(f"  Lambda: {lambda_now:.6f}")
            
            # === SAMPLE CLASSIFICATION DIAGNOSTIC (MODIFIED TO DUAL SAMPLE) ===
            # The existing check is reused to comply with 'no delete'
            if sample_logits_cpu is not None and sample_label_cpu is not None:
                xm.master_print("\nDUAL SAMPLE CLASSIFICATION DIAGNOSTIC (One Positive & One Negative Sample):")
                
                # Print Positive Sample Diagnostic
                pos_output = format_diagnostic_output(sample_logits_pos_cpu, sample_label_pos_cpu, "Positive (Label 1)")
                xm.master_print(pos_output)

                # Print Negative Sample Diagnostic
                neg_output = format_diagnostic_output(sample_logits_neg_cpu, sample_label_neg_cpu, "Negative (Label 0)")
                xm.master_print(neg_output)
            
            xm.master_print("")
            xm.master_print("WEIGHT SYNCHRONIZATION CHECK:")
            for i, (param_sum, param_element) in enumerate(param_checks):
                expected_sum = param_element * num_cores
                diff = param_sum - expected_sum
                xm.master_print(f"  Parameter {i}: Difference = {diff}")
            xm.master_print("-" * 80)
        
        # --- Checkpointing (Same as before) ---
        if (epoch + 1) % flags["checkpoint_interval"] == 0:
            if rank == 0:
                checkpoint_path = f"/tmp/controller_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                xm.master_print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Ensure all cores finish the epoch before proceeding
        xm.rendezvous(f"epoch_end_{epoch}")

    # --- FINAL SUMMARY ---
    total_time = time.time() - start_time
    if rank == 0:
        xm.master_print("\n" + "=" * 80)
        xm.master_print("TRAINING FINISHED")
        xm.master_print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        xm.master_print(f"Total steps: {global_step}")
        xm.master_print(f"✅ TRAINING COMPLETED ON ALL CORES. Active cores: {num_cores}/32")
        xm.master_print("=" * 80)
    
    xm.rendezvous("final_check")

# ... (Rest of your _mp_fn and __main__ block) ...
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
        "batch_size": 64, # Smaller batch size for 19500 samples
        
        # Halting loss schedule, halting loss should at first be very small then gradually go to a maximum where it matters about exactly as much as CLS
        "lambda_start": 0.0001,
        "lambda_target": 0.01,
        
        # Training, leave at 5 if model seems to be converging, else go to 10
        "epochs": 5,
        
        # Data loading
        "chunk_filename": "embeddings_chunk_0.npz", # Change to desired chunk
        "samples_per_shard": 19500, # Number of samples per core
        
        # Logging and checkpointing
        "log_interval": 50, # Log every N steps
        "checkpoint_interval": 1, # Save checkpoint every N epochs
    }
    
    # Automatically detect number of TPU cores
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method='fork')