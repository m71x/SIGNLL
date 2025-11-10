import os
import json
import numpy as np
import torch
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# THIS CODE EXTRACTS THE CLS TOKEN EMBEDDING FROM EVERY LAYER AND THE FINAL PREDICTION.
# --- Configuration (Added Efficiency Parameters) ---
BUCKET_NAME = "encoder-models"
MODEL_PREFIX = "siebert"
LOCAL_MODEL_PATH = "/home/mikexi/siebert_model"
UPLOAD_PREFIX = "siebert-data/siebert-data-test"

# --- Efficiency Parameters ---
# Number of samples to process in a single TPU forward pass (must be a factor of the dataset size)
INFERENCE_BATCH_SIZE = 64
# Number of *samples* to accumulate in memory before writing one large .npz file and uploading
# Using 10000 samples for better I/O amortization
IO_ACCUMULATION_THRESHOLD = 10000 
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
        output_hidden_states=True # Keep this to get all layer outputs
    ).to(device)
    model.eval()
    
    # DETERMINE MAX LENGTH HERE
    # Use the model's max length, defaulting to 512 if not explicitly set
    MAX_SEQ_LENGTH = tokenizer.model_max_length if tokenizer.model_max_length > 0 and tokenizer.model_max_length <= 512 else 512
    print(f"[Core {core_id}] Using fixed max sequence length: {MAX_SEQ_LENGTH}", flush=True)
    
    # --- Step 3 (MODIFIED): Setup I/O Accumulation ---
    
    # The list will hold the hidden state outputs (NumPy arrays) for many batches
    accumulated_cls_outputs = [] # New list for CLS tokens
    accumulated_predictions = [] # New list for final 0/1 predictions
    
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
        
        # Tokenization with fixed max_length padding
        inputs = tokenizer(
            sample_texts, 
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
        # --- B. Data Transfer and Accumulation (Host CPU Memory) ---
        
        # New: Collect all CLS tokens from ALL layers
        all_layer_cls_tokens = [
            t[:, 0, :].to(torch.float32).cpu().numpy() # [:, 0, :] extracts the CLS token
            for t in outputs.hidden_states
        ]
        
        # Stack the CLS tokens from all layers together into one array (Batch, Num_Layers, Dim)
        # The embedding output and N layers gives N+1 layers, usually 13 arrays for BERT-base
        stacked_cls_tokens = np.stack(all_layer_cls_tokens, axis=1) # Shape: (Batch_Size, Num_Layers, Hidden_Dim)
        accumulated_cls_outputs.append(stacked_cls_tokens)
        
        # New: Extract the final prediction (0 or 1)
        # Logits are (Batch, 2). Argmax gives the predicted class index (0 or 1).
        batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy() # Shape: (Batch_Size,)
        accumulated_predictions.append(batch_predictions)
        
        current_io_sample_count += INFERENCE_BATCH_SIZE
        
        print(f"[Core {core_id}] Processed batch {batch_i+1}/{num_batches}. Buffered samples: {current_io_sample_count}", flush=True)

        
        # --- C. I/O Write-Out Logic (Amortizing the cost) ---
        # Check if the accumulated data hits the efficiency threshold
        if current_io_sample_count >= IO_ACCUMULATION_THRESHOLD:
            
            print(f"[Core {core_id}] 🔥 I/O THRESHOLD REACHED. Writing chunk {chunk_index}...", flush=True)

            # 4. Save locally using NumPy (one large file)
            local_path = os.path.join(local_dir, f"embeddings_chunk_{chunk_index}.npz")
            
            # Concatenate CLS tokens from all batches
            all_cls_tokens = np.concatenate(accumulated_cls_outputs, axis=0) # Shape: (Total_Samples, Num_Layers, Hidden_Dim)
            
            # Concatenate predictions from all batches
            all_predictions = np.concatenate(accumulated_predictions, axis=0) # Shape: (Total_Samples,)

            # Save both arrays to the NPZ file
            np.savez_compressed(
                local_path, 
                cls_embeddings=all_cls_tokens, 
                predictions=all_predictions.astype(np.uint8) # Save predictions as efficient integers
            )
            
            print(f"[Core {core_id}] Saved {all_cls_tokens.shape[0]} samples locally to {local_path}", flush=True)

            # 5. Upload file to GCS
            gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/embeddings_chunk_{chunk_index}.npz"
            upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
            
            # Reset the accumulators
            accumulated_cls_outputs = []
            accumulated_predictions = []
            current_io_sample_count = 0
            chunk_index += 1
            
            # Optional: Clean up the local file immediately after upload
            os.remove(local_path)
            
            print(f"[Core {core_id}] Chunk {chunk_index-1} complete. Time taken: {time.time() - start_time:.2f} seconds.", flush=True)

    # --- Step 7: Final Flush (Flush any remaining data) ---
    if accumulated_cls_outputs:
        print(f"[Core {core_id}] 🧊 Final flush of {current_io_sample_count} samples.", flush=True)
        
        local_path = os.path.join(local_dir, f"embeddings_chunk_final_{chunk_index}.npz")
        
        # Final concatenation
        all_cls_tokens = np.concatenate(accumulated_cls_outputs, axis=0)
        all_predictions = np.concatenate(accumulated_predictions, axis=0)
        
        np.savez_compressed(
            local_path, 
            cls_embeddings=all_cls_tokens, 
            predictions=all_predictions.astype(np.uint8)
        )

        gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/embeddings_chunk_final_{chunk_index}.npz"
        upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
        os.remove(local_path)


    # --- Step 8: Final Rendezvous and Completion ---
    xm.rendezvous("done")
    end_time = time.time()
    print(f"[Core {core_id}] Test complete ✅. Total duration for all samples: {end_time - start_time:.2f} seconds.", flush=True)


# --- Entry point (Unchanged) ---
if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())