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

# ====================================================================
# GCS Access Code (Integrated from gcs_access.py)
# ====================================================================
from google.cloud import storage

# --- Configuration Constants (for GCS) ---
BUCKET_NAME = "encoder-models-2"
UPLOAD_PREFIX = "result-models"
FINAL_MODEL_DIR = "/tmp/final_controller_model"

# --- Helper: Upload file to GCS ---
def upload_file_to_gcs(bucket_name, local_path, gcs_blob_path, max_retries=3):
    """
    Uploads a file from local storage to a GCS blob with retry logic and verification.
    (Content of function is as provided)
    """
    import time
    
    for attempt in range(max_retries):
        try:
            # CRITICAL: Check if file still exists before retry
            if not os.path.exists(local_path):
                print(f"❌ Attempt {attempt+1}: Local file no longer exists: {local_path}", flush=True)
                return False
            
            # Get local file size for verification
            local_size = os.path.getsize(local_path)
            print(f"🔄 Attempt {attempt+1}: Uploading {local_size:,} bytes from {local_path}", flush=True)
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_blob_path)
            
            # Upload with a timeout
            blob.upload_from_filename(local_path, timeout=300)
            
            # CRITICAL: Reload and verify
            blob.reload()
            
            # Verify size match
            if blob.size != local_size:
                print(f"❌ Attempt {attempt+1}: Size mismatch! Local: {local_size}, GCS: {blob.size}", flush=True)
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {2**attempt} seconds before retry...", flush=True)
                    time.sleep(2 ** attempt)
                    continue
                return False
            
            # Verify blob exists and is readable
            if not blob.exists():
                print(f"❌ Attempt {attempt+1}: Blob does not exist after upload!", flush=True)
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {2**attempt} seconds before retry...", flush=True)
                    time.sleep(2 ** attempt)
                    continue
                return False
                
            print(f"✅ Uploaded & Verified: {local_path} → gs://{bucket_name}/{gcs_blob_path} ({blob.size:,} bytes)", flush=True)
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed for {local_path}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                # Check if file still exists before retrying
                if os.path.exists(local_path):
                    print(f"⏳ File still exists. Waiting {2**attempt} seconds before retry...", flush=True)
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ File disappeared! Cannot retry.", flush=True)
                    return False
            else:
                return False
    
    return False

# --- Helper: Upload model directory to GCS (NEW) ---
def upload_model_to_gcs(bucket_name, local_dir, gcs_prefix):
    """
    Uploads all files in a local directory to a GCS prefix.
    """
    print(f"Starting model upload from {local_dir} to gs://{bucket_name}/{gcs_prefix}...", flush=True)
    success = True
    
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(root, file)
            # Create a path relative to the local_dir
            rel_path = os.path.relpath(local_file, local_dir)
            gcs_blob_path = os.path.join(gcs_prefix, rel_path)
            
            if not upload_file_to_gcs(bucket_name, local_file, gcs_blob_path):
                print(f"🚨 Failed to upload critical file: {local_file}. Aborting model upload.", flush=True)
                success = False
                break
        if not success:
            break
            
    if success:
        print(f"✅ Final model upload to GCS prefix gs://{bucket_name}/{gcs_prefix} complete!", flush=True)
    return success

# ====================================================================
# Training Logic (train_loop)
# ====================================================================

# NOTE: The train_loop is modified to accept model and optimizer and handle a single chunk.
def train_loop(rank, flags, model, optimizer, current_chunk_idx):
    
    device = xm.torch_xla.device()
    num_cores = xm.xrt_world_size()

    # --- Load Data Once (On CPU!) ---
    # Each core loads its own shard
    
    # Use the specific chunk index passed to the function
    current_filename = f"embeddings_chunk_{current_chunk_idx}.npz"
    flags["chunk_filename"] = current_filename # Update flag for logging/download function
    
    if rank == 0:
        xm.master_print(f"\n{'#'*80}")
        xm.master_print(f"🚀 Starting TRAINING on CHUNK {current_chunk_idx} ({current_filename})")
        xm.master_print(f"{'#'*80}")

    data = training_data_download(
        core_id=rank,
        filename=current_filename,
        max_entries=flags["samples_per_shard"]
    )
    
    if data is None:
        raise RuntimeError(f"[Core {rank}] Failed to load training data for chunk {current_chunk_idx}")

    # CRITICAL: Keep data on CPU. DO NOT call .to(device) here.
    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    # Handle layer slicing (still on CPU)
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]
    elif teacher_cls_full.shape[1] != 24:
        raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")

    # === Calculate and Apply Class Weighting ===
    # Using .sum() returns a scalar tensor, which is safe on CPU before pos_weight_value calculation.
    # *** CORRECTION: Removed .item() ***
    neg_samples_count = (teacher_label_full == 0).sum() 
    pos_samples_count = (teacher_label_full == 1).sum() 
    
    # pos_weight_value is a float calculated on the CPU
    pos_weight_value = neg_samples_count / pos_samples_count 
    pos_weight_tensor = torch.tensor([pos_weight_value]).float()
    
    if rank == 0:
        # Use .item() *only* for the print statement on the master core
        xm.master_print(f"Chunk {current_chunk_idx} Data Shard Check: Neg={neg_samples_count.item()}, Pos={pos_samples_count.item()}")
        xm.master_print(f"Calculated Positive Weight (pos_weight): {pos_weight_value.item():.4f}")
    
    # CRITICAL: Move pos_weight to device before passing to loss function
    pos_weight_device = pos_weight_tensor.to(device)
    L = 24

    # Instantiate BCE loss with the calculated weight (pos_weight)
    bce_loss_fn = nn.BCEWithLogitsLoss(
        reduction="none", 
        pos_weight=pos_weight_device
    ).to(device)
    
    # --- Create Dataset and DataLoader ---
    num_samples = teacher_cls_full.shape[0] # Used for logging only
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

    xm.rendezvous("chunk_data_prepared")

    # --- Training Configuration ---
    num_epochs = flags["epochs"]
    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
    num_batches_per_epoch = len(data_loader)
    total_steps = num_epochs * num_batches_per_epoch

    if rank == 0:
        xm.master_print(f"Chunk {current_chunk_idx} training config: {num_epochs} epochs, {num_batches_per_epoch} batches/epoch")
    
    global_step = 0
    start_time = time.time()
    model.train()
    
    # ... (Diagnostic variables remain the same) ...
    sample_logits_cpu = None
    sample_label_cpu = None
    sample_logits_pos_cpu = None
    sample_label_pos_cpu = None
    sample_logits_neg_cpu = None
    sample_label_neg_cpu = None
    
    # --- (4) Full Training Loop (Refactored) ---
    for epoch in range(num_epochs):
        if rank == 0:
            xm.master_print(f"\n{'='*80}")
            xm.master_print(f"CHUNK {current_chunk_idx} EPOCH {epoch + 1}/{num_epochs}")
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
                sample_logits_cpu = class_logits[0].detach().cpu()
                sample_label_cpu = teacher_label[0].detach().cpu()
                
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
            
            # Halting loss (re-enabled)
            depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
            halt_penalty = (depths * (1 - h)).sum(dim=1)
            
            # Calculate total steps remaining (for the whole process across all chunks)
            total_chunks = flags["total_chunks"]
            steps_per_chunk = num_epochs * len(data_loader)
            
            # Global step for lambda calculation needs to be tracked across chunks
            global_step_total = (current_chunk_idx * steps_per_chunk) + global_step
            total_steps_all = total_chunks * steps_per_chunk
            
            progress = global_step_total / total_steps_all
            lambda_now = lambda_start + (lambda_target - lambda_start) * progress
            loss_halt = lambda_now * halt_penalty.mean()
            
            # Total loss
            loss = loss_cls + loss_halt
            
            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            xm.mark_step()
            
            global_step += 1
        
        # --- End of Epoch Diagnostics (Same as before) ---
        loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
        loss_cls_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
        loss_halt_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
        h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
        
        param_checks = []
        for i, param in enumerate(model.parameters()):
            if i >= 3: break
            param_element = param.flatten()[0]
            param_sum = xm.all_reduce(xm.REDUCE_SUM, param_element)
            param_checks.append((param_sum, param_element))
        
        xm.mark_step()
        
        if rank == 0:
            
            def format_diagnostic_output(logits_cpu, label_cpu, sample_type):
                if logits_cpu is None or label_cpu is None:
                    return f"  No {sample_type} sample found in the first batch of this core."
                sample_probs = torch.softmax(logits_cpu, dim=-1)
                predicted_classes = torch.argmax(sample_probs, dim=-1).tolist()
                max_confidences = sample_probs.max(dim=-1).values
                predicted_probs = [f"{p:.4f}" for p in max_confidences.tolist()]
                true_label_value = label_cpu.item()
                
                output = []
                output.append(f"  --- {sample_type} (Chunk {current_chunk_idx}, True Label: {true_label_value}) ---")
                output.append(f"  Predicted Class per Layer (0-23): {predicted_classes}")
                output.append(f"  Max Confidence per Layer: {predicted_probs}")
                return "\n".join(output)
            
            elapsed = time.time() - start_time
            xm.master_print("-" * 80)
            xm.master_print(f"CHUNK {current_chunk_idx} EPOCH {epoch+1}/{num_epochs} COMPLETED")
            xm.master_print(f"  Total Elapsed Time: {elapsed:.1f}s")
            xm.master_print("FINAL METRICS:")
            xm.master_print(f"  Step (Chunk): {global_step}/{total_steps}")
            xm.master_print(f"  Global Step (Total): {global_step_total}/{total_steps_all}")
            xm.master_print(f"  Avg Total Loss: {loss_sum / num_cores}")
            xm.master_print(f"  Avg Cls Loss: {loss_cls_sum / num_cores}")
            xm.master_print(f"  Avg Halt Loss: {loss_halt_sum / num_cores}")
            xm.master_print(f"  Avg Mean h: {h_mean / num_cores}")
            xm.master_print(f"  Lambda: {lambda_now:.6f}")
            
            if sample_logits_pos_cpu is not None and sample_logits_neg_cpu is not None:
                xm.master_print("\nDUAL SAMPLE CLASSIFICATION DIAGNOSTIC:")
                pos_output = format_diagnostic_output(sample_logits_pos_cpu, sample_label_pos_cpu, "Positive (Label 1)")
                xm.master_print(pos_output)
                neg_output = format_diagnostic_output(sample_logits_neg_cpu, sample_label_neg_cpu, "Negative (Label 0)")
                xm.master_print(neg_output)
            
            # --- Checkpointing (Modified to use FINAL_MODEL_DIR for final state) ---
            checkpoint_dir = FINAL_MODEL_DIR
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"controller_chunk_{current_chunk_idx}_epoch_{epoch+1}.pt")
            
            # Save the full state (model and optimizer)
            torch.save({
                'epoch': epoch + 1,
                'chunk': current_chunk_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            xm.master_print(f"✅ Checkpoint saved: {checkpoint_path}")
            
        xm.rendezvous(f"chunk_{current_chunk_idx}_epoch_end_{epoch}")

    return model, optimizer


# ====================================================================
# Main Execution Block
# ====================================================================

def _mp_fn(rank, flags):
    try:
        torch.set_default_tensor_type('torch.FloatTensor')
        
        device = xm.torch_xla.device()
        L = 24 # Number of layers
        
        # --- 1. Initialize Model and Optimizer (outside the loop) ---
        model = Controller(
            L=L,
            d_teacher=1024,
            d_ctrl=flags["d_ctrl"],
            n_layers=flags["transformer_layers"],
            num_classes=2
        ).to(device)
        
        # Synchronize initial weights (Only needs to happen once)
        if rank == 0:
            xm.master_print("Synchronizing initial weights across all cores...")
        for param in model.parameters():
            if rank == 0:
                param.data = param.data * xm.xrt_world_size()
            else:
                param.data = param.data * 0
            param.data = xm.all_reduce(xm.REDUCE_SUM, param.data)
            param.data = param.data / xm.xrt_world_size()
        xm.mark_step()
        xm.rendezvous("initial_weights_synced")
        if rank == 0:
            xm.master_print("✅ Initial weights synchronized.")

        optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)

        # --- 2. Iterate through all chunks (0 to 28) ---
        for chunk_idx in range(flags["start_chunk"], flags["end_chunk"] + 1):
            # Pass the model and optimizer into the train_loop for continuous training
            model, optimizer = train_loop(rank, flags, model, optimizer, chunk_idx)
            xm.rendezvous(f"chunk_completed_{chunk_idx}")
            
        # --- 3. Final Model Export (Master Only) ---
        if rank == 0:
            xm.master_print("\n" + "*" * 80)
            xm.master_print("ALL CHUNKS COMPLETE. STARTING FINAL MODEL EXPORT.")
            
            # Create a simple file to signify training completion and save only the model state
            final_save_path = os.path.join(FINAL_MODEL_DIR, "final_model_weights.pt")
            torch.save(model.state_dict(), final_save_path)
            xm.master_print(f"✅ Final model weights saved locally to: {final_save_path}")
            
            # Upload the entire directory to GCS
            gcs_final_path = os.path.join(UPLOAD_PREFIX, "final_controller")
            upload_model_to_gcs(BUCKET_NAME, FINAL_MODEL_DIR, gcs_final_path)

            xm.master_print("✨ TRAINING AND EXPORT COMPLETE. ✨")
            xm.master_print("*" * 80)

        # Final rendezvous
        xm.rendezvous("final_run_completed")

    except Exception as e:
        if rank == 0:
            print(f"[Core {rank}] FATAL ERROR: {e}", flush=True)
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
        "batch_size": 64,
        
        # Halting loss schedule (Optimal from previous runs)
        "lambda_start": 0.0001,
        "lambda_target": 0.025,
        
        # Training
        "epochs": 10,
        
        # Data loading and looping
        "start_chunk": 0,
        "end_chunk": 28,
        "total_chunks": 29,
        "samples_per_shard": 19500,
        
        # Logging and checkpointing
        "log_interval": 50,
        "checkpoint_interval": 1, 
    }
    
    # The actual chunk filename is now set inside train_loop
    FLAGS["chunk_filename"] = "" 
    
    # Automatically detect number of TPU cores
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method='fork')
