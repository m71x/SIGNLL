import os
import json
import numpy as np
import torch
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time # Included time for completeness, though not strictly required by this prompt


# --- Configuration ---
BUCKET_NAME = "encoder-models-2"
MODEL_PREFIX = "siebert"
LOCAL_MODEL_PATH = "/home/mikexi/siebert_model"
UPLOAD_PREFIX = "siebert-data/siebert-data-test"


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


# --- Helper: Download file from GCS ---
def download_file_from_gcs(bucket_name, gcs_blob_path, local_path):
    """Downloads a file from GCS to local storage."""
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


# --- Worker function for each TPU core ---
def _mp_fn(index):
    # Use the non-deprecated way to get ordinal for future compatibility
    core_id = xmp.get_global_ordinal() if hasattr(xmp, 'get_global_ordinal') else xm.get_ordinal()
    device = xm.torch_xla.device()

    print(f"[Core {core_id}] Using device: {device}", flush=True)

    # --- Step 1: Only core 0 downloads model ---
    if xm.is_master_ordinal():
        if not os.path.exists(LOCAL_MODEL_PATH):
            download_model_from_gcs(BUCKET_NAME, MODEL_PREFIX, LOCAL_MODEL_PATH)
        else:
            print("[Core 0] Model already cached locally.", flush=True)

    xm.rendezvous("model_ready")

    # --- Step 2: Load model & tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH,
        # Corrected: Use 'dtype' instead of deprecated 'torch_dtype'
        dtype=torch.bfloat16, 
        output_hidden_states=True
    ).to(device)
    model.eval()

    # --- Step 3: Run inference and collect hidden states ---
    sample_text = f"This is a SieBERT hidden state test from core {core_id}."
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Cast to float32 before moving to CPU/NumPy for stability
        hidden_states = [t.to(torch.float32).cpu().numpy() for t in outputs.hidden_states] 
        logits = outputs.logits.to(torch.float32).cpu().numpy()
        hidden_states.append(logits)

    print(f"[Core {core_id}] ✅ Appended classification output as final hidden entry.", flush=True)

    # --- Step 4: Save locally using NumPy ---
    local_dir = f"/tmp/siebert_core_{core_id}"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "hidden_states.npz")
    np.savez_compressed(local_path, *hidden_states)
    print(f"[Core {core_id}] Saved NumPy data locally to {local_path}", flush=True)

    # --- Step 5: Upload file to GCS ---
    gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/hidden_states.npz"
    upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
    
    # --- Step 6 (MODIFIED): Download, Load, and Print Contents from GCS ---
    
    # Path where the downloaded file will be saved locally
    download_path = os.path.join(local_dir, "hidden_states_downloaded.npz")
    
    # Download the file from GCS
    if download_file_from_gcs(BUCKET_NAME, gcs_blob_path, download_path):
        try:
            # Load the .npz file
            reloaded_npz = np.load(download_path)
            
            # Print information about the loaded arrays
            print(f"[Core {core_id}] ----------------------------------------", flush=True)
            print(f"[Core {core_id}] 📥 Loaded .npz file from GCS successfully. Contents:", flush=True)
            
            # Iterate through the named arrays in the .npz file
            for i, key in enumerate(reloaded_npz.files):
                array = reloaded_npz[key]
                print(f"[Core {core_id}]   Array {i+1} ('{key}'): Shape={array.shape}, DType={array.dtype}", flush=True)
                
                # Print the first few elements of the first array (Hidden State Layer 0)
                if i == 0:
                    print(f"[Core {core_id}]   First elements of Array 1 (Hidden State 0):\n{array[0, 0, :5]}...", flush=True)
                # Print the content of the final array (Logits)
                elif i == len(reloaded_npz.files) - 1:
                    print(f"[Core {core_id}]   Final array (Logits) content:\n{array}", flush=True)

            print(f"[Core {core_id}] ----------------------------------------", flush=True)
            
        except Exception as e:
            print(f"[Core {core_id}] ❌ Failed to load or print .npz contents: {e}", flush=True)


    xm.rendezvous("done")
    print(f"[Core {core_id}] Test complete ✅", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())