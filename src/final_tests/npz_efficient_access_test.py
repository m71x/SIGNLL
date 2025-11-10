import os
import json
import numpy as np
import torch
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

#THIS CODE CHECKS THAT A MASSIVE AMOUNT OF COLLECTIONS OF ALL HIDDEN STATE VECTORS CAN BE UPLOADED TO GCS. IN PRACTICE, YOU ONLY NEED THE CLS TOKEN
# --- Configuration (Added Efficiency Parameters) ---
BUCKET_NAME = "encoder-models"
MODEL_PREFIX = "siebert"
LOCAL_MODEL_PATH = "/home/mikexi/siebert_model"
UPLOAD_PREFIX = "siebert-data/siebert-data-test"

# --- Efficiency Parameters ---
# Number of samples to process in a single TPU forward pass (must be a factor of the dataset size)
INFERENCE_BATCH_SIZE = 64
# Number of *samples* to accumulate in memory before writing one large .npz file and uploading
IO_ACCUMULATION_THRESHOLD = 5000 
# Total number of samples this core will process (Placeholder for a real dataset)
TOTAL_SAMPLES_TO_PROCESS = 10000 

# --- Helper: Upload file to GCS ---
def upload_file_to_gcs(bucket_name, local_path, gcs_blob_path):
    """Uploads a file to GCS (disk-based)."""
    core_id = xm.get_ordinal()
    print(f"[Core {core_id}] Starting GCS upload for {local_path}...", flush=True)
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_path)
        blob.upload_from_filename(local_path)
        print(f"[Core {core_id}] ✅ Uploaded: {local_path} → gs://{bucket_name}/{gcs_blob_path}", flush=True)
        return True
    except Exception as e:
        print(f"[Core {core_id}] ❌ Upload failed for {local_path}: {e}", flush=True)
        return False


# --- Helper: Download file from GCS (Kept for completeness, but not used in the efficient pipeline) ---
def download_file_from_gcs(bucket_name, gcs_blob_path, local_path):
    """Downloads a file from GCS to local storage."""
    # ... (body remains the same)
    core_id = xm.get_ordinal()
    print(f"[Core {core_id}] Starting GCS download for {gcs_blob_path}...", flush=True)
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"[Core {core_id}] ✅ Downloaded: gs://{bucket_name}/{gcs_blob_path} → {local_path}", flush=True)
        return True
    except Exception as e:
        print(f"[Core {core_id}] ❌ Download failed for {gcs_blob_path}: {e}", flush=True)
        return False


# --- Model download (only by master) ---
def download_model_from_gcs(bucket_name, prefix, local_dir):
    # ... (body remains the same)
    print(f"[Core 0] Downloading model from gs://{bucket_name}/{prefix} ...", flush=True)
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    count = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel_path = blob.name[len(prefix):].lstrip("/")
        local_file = f"{local_dir}/{rel_path}"
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)
        count += 1
    print(f"[Core 0] Model download complete ({count} files).", flush=True)


# --- Worker function for each TPU core (EFFICIENTLY MODIFIED) ---
def _mp_fn(index):
    core_id = xmp.get_global_ordinal() if hasattr(xmp, 'get_global_ordinal') else xm.get_ordinal()
    device = xm.torch_xla.device()
    local_dir = f"/tmp/siebert_core_{core_id}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[Core {core_id}] Using device: {device}", flush=True)

    # --- Step 1 & 2: Model Setup (Unchanged) ---
    if xm.is_master_ordinal():
        if not os.path.exists(LOCAL_MODEL_PATH):
            download_model_from_gcs(BUCKET_NAME, MODEL_PREFIX, LOCAL_MODEL_PATH)
        else:
            print("[Core 0] Model already cached locally.", flush=True)

    xm.rendezvous("model_ready")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH,
        dtype=torch.bfloat16, 
        output_hidden_states=True
    ).to(device)
    model.eval()
    
    # DETERMINE MAX LENGTH HERE
    # Use the model's max length, defaulting to 512 if not explicitly set
    MAX_SEQ_LENGTH = tokenizer.model_max_length if tokenizer.model_max_length > 0 and tokenizer.model_max_length <= 512 else 512
    print(f"[Core {core_id}] Using fixed max sequence length: {MAX_SEQ_LENGTH}", flush=True)
    
    # --- Step 3 (MODIFIED): Setup I/O Accumulation ---
    
    # The list will hold the hidden state outputs (NumPy arrays) for many batches
    accumulated_outputs = []
    
    # Counter for the current number of samples buffered in memory
    current_io_sample_count = 0
    
    # Counter for the file chunks saved by this core
    chunk_index = 0
    
    # --- Simulated Data Iteration (Replace with actual data loader later) ---
    num_batches = TOTAL_SAMPLES_TO_PROCESS // INFERENCE_BATCH_SIZE
    
    start_time = time.time()
    
    for batch_i in range(num_batches):
        
        # --- A. Inference Batching (Simulated input batch) ---
        
        # Create a list of text samples for a large batch
        sample_texts = [
            f"This is sample {batch_i * INFERENCE_BATCH_SIZE + i} for core {core_id}. This text is intentionally a bit longer to test the max length padding." 
            for i in range(INFERENCE_BATCH_SIZE)
        ]
        
        # FIX: Tokenization with fixed max_length padding
        inputs = tokenizer(
            sample_texts, 
            padding='max_length', # <--- FIXED: Ensure all samples are padded to MAX_SEQ_LENGTH
            truncation=True,
            max_length=MAX_SEQ_LENGTH, # <--- FIXED: Specify the maximum length
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
        # --- B. Data Transfer and Accumulation (Host CPU Memory) ---
        
        # Collect all hidden state tensors
        batch_hidden_states = [t.to(torch.float32).cpu().numpy() for t in outputs.hidden_states]
        batch_logits = outputs.logits.to(torch.float32).cpu().numpy()
        
        # The hidden states now all have the same sequence dimension (MAX_SEQ_LENGTH), 
        # so concatenation later will not fail.
        final_layer_hidden_state = batch_hidden_states[-1] 
        
        # Accumulate the final layer hidden states and logits for the entire batch
        accumulated_outputs.append(final_layer_hidden_state)
        accumulated_outputs.append(batch_logits)
        
        current_io_sample_count += INFERENCE_BATCH_SIZE
        
        print(f"[Core {core_id}] Processed batch {batch_i+1}/{num_batches}. Buffered samples: {current_io_sample_count}", flush=True)

        
        # --- C. I/O Write-Out Logic (Amortizing the cost) ---
        # Check if the accumulated data hits the efficiency threshold
        if current_io_sample_count >= IO_ACCUMULATION_THRESHOLD:
            
            print(f"[Core {core_id}] 🔥 I/O THRESHOLD REACHED. Writing chunk {chunk_index}...", flush=True)

            # 4. Save locally using NumPy (one large file)
            local_path = os.path.join(local_dir, f"hidden_states_chunk_{chunk_index}.npz")
            
            # Concatenate all accumulated arrays. This now works because all hidden state arrays
            # have the same shape (BATCH_SIZE, MAX_SEQ_LENGTH, 1024) and all logits arrays have 
            # the same shape (BATCH_SIZE, 2).
            all_hidden_states = np.concatenate([arr for arr in accumulated_outputs if arr.ndim == 3], axis=0) # ndim=3 for (Batch, Seq, Dim)
            all_logits = np.concatenate([arr for arr in accumulated_outputs if arr.ndim == 2 and arr.shape[-1] == 2], axis=0) # ndim=2 for (Batch, Logits)

            np.savez_compressed(local_path, hidden_states=all_hidden_states, logits=all_logits)
            
            print(f"[Core {core_id}] Saved {all_hidden_states.shape[0]} samples locally to {local_path}", flush=True)

            # 5. Upload file to GCS
            gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/hidden_states_chunk_{chunk_index}.npz"
            upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
            
            # Reset the accumulator
            accumulated_outputs = []
            current_io_sample_count = 0
            chunk_index += 1
            
            # Optional: Clean up the local file immediately after upload
            os.remove(local_path)
            
            print(f"[Core {core_id}] Chunk {chunk_index-1} complete. Time taken: {time.time() - start_time:.2f} seconds.", flush=True)

    # --- Step 7: Final Flush (Flush any remaining data) ---
    if accumulated_outputs:
        print(f"[Core {core_id}] 🧊 Final flush of {current_io_sample_count} samples.", flush=True)
        
        local_path = os.path.join(local_dir, f"hidden_states_chunk_final_{chunk_index}.npz")
        
        # Final concatenation uses the fixed shape logic
        all_hidden_states = np.concatenate([arr for arr in accumulated_outputs if arr.ndim == 3], axis=0)
        all_logits = np.concatenate([arr for arr in accumulated_outputs if arr.ndim == 2 and arr.shape[-1] == 2], axis=0)
        np.savez_compressed(local_path, hidden_states=all_hidden_states, logits=all_logits)

        gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/hidden_states_chunk_final_{chunk_index}.npz"
        upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
        os.remove(local_path)


    # --- Step 8: Final Rendezvous and Completion ---
    xm.rendezvous("done")
    end_time = time.time()
    print(f"[Core {core_id}] Test complete ✅. Total duration for all samples: {end_time - start_time:.2f} seconds.", flush=True)


# --- Entry point (Unchanged) ---
if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())