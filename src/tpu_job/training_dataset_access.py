import os
import json
from google.cloud import storage
import gcld3
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor # New import for parallel CPU work


# Attempt to import XLA for distributed logging. Use a fallback if not available.
try:
    import torch_xla.core.xla_model as xm
    def get_core_ordinal():
        """Returns the XLA core ID for distributed logging."""
        return xm.get_ordinal()
except ImportError:
    # Fallback for local testing/non-TPU environments
    def get_core_ordinal():
        """Fallback function when torch_xla is not imported."""
        return 0 

# --- Configuration Constants ---
BUCKET_NAME = "encoder-models-2"
GCS_DATA_PREFIX = "twitter-100m"
GCS_CHECKPOINT_PREFIX = "tensorcore-checkpoints-v2-init"
CHECKPOINT_FILENAME = "progress.json"
# The core ID (0-63 for a v3-64 pod) is used directly as the shard_index
PARQUET_FILE_FORMAT = "tweets-{shard_index:02d}-of-64.parquet"

GC_DETECTOR = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
# Define the number of CPU workers for parallel pre-processing (tune this based on your machine's CPU cores)
NUM_CPU_WORKERS = 8 

def check_language(text):
    """Returns 1 if English and reliable, 0 otherwise."""
    if not isinstance(text, str):
        return 0
    try:
        # NOTE: Using the global GC_DETECTOR
        result = GC_DETECTOR.FindLanguage(text=text)
        # 1 for English/reliable, 0 otherwise
        return 1 if result.is_reliable and result.language == 'en' else 0
    except Exception:
        # Catch any exceptions during language check
        return 0
# --- GCS Helper Functions ---

def _get_gcs_client_and_blob(bucket_name, gcs_blob_path):
    """Initializes GCS client, bucket, and blob object."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    return blob


def download_file_from_gcs(local_path, bucket_name, gcs_blob_path):
    """
    Downloads a file from GCS to local storage, handling path creation 
    and logging errors.

    Args:
        local_path (str): The local path where the file should be saved.
        bucket_name (str): The GCS bucket name.
        gcs_blob_path (str): The full path to the blob in GCS (e.g., 'prefix/file.ext').

    Returns:
        bool: True if download was successful, False otherwise.
    """
    core_id = get_core_ordinal()
    try:
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        
        blob = _get_gcs_client_and_blob(bucket_name, gcs_blob_path)
        
        if not blob.exists():
             print(f"[Core {core_id}] ⚠️ GCS blob not found: gs://{bucket_name}/{gcs_blob_path}", flush=True)
             return False

        blob.download_to_filename(local_path)
        print(f"[Core {core_id}] ✅ Downloaded: gs://{bucket_name}/{gcs_blob_path} → {local_path}", flush=True)
        return True
    except Exception as e:
        print(f"[Core {core_id}] ❌ Download failed for {gcs_blob_path}: {e}", flush=True)
        return False


# --- Dataset Shard Access Functions ---

def get_shard_path(core_id, local_dir="/tmp/data"):
    """
    Generates the expected local file path for the data shard assigned 
    to a given TPU core ID.

    Args:
        core_id (int): The ID of the current TPU core (which is used as the shard index).
        local_dir (str): The local temporary directory where data shards are stored.

    Returns:
        str: The full local path for the data shard file.
    """
    # Parquet filename format uses the core_id directly as the shard_index
    filename = PARQUET_FILE_FORMAT.format(shard_index=core_id)
    local_path = os.path.join(local_dir, filename)
    return local_path

def download_data_shard(core_id, local_dir, bucket_name, gcs_prefix, parquet_format):
    """
    Downloads the data shard for a given core, processes it to add the 
    language column, and returns the local path to the processed file.
    """
    core_id_log = get_core_ordinal()
    shard_filename = parquet_format.format(shard_index=core_id)
    gcs_blob_path = f"{gcs_prefix}/{shard_filename}"
    local_path = os.path.join(local_dir, shard_filename)

    # 1. Download the file from GCS
    print(f"[Core {core_id_log}] Attempting to download data shard {shard_filename} to {local_path}...", flush=True)
    if not download_file_from_gcs(local_path, bucket_name, gcs_blob_path):
        return None # Return None if download failed

    # --- NEW: Process the file with Parallel Language Detection ---
    print(f"[Core {core_id_log}] Loading parquet file from {local_path}...", flush=True)
    try:
        df = pd.read_parquet(local_path)
    except Exception as e:
        print(f"[Core {core_id_log}] ❌ Error loading parquet file: {e}. Cannot proceed with pre-processing.", flush=True)
        return None

    # Use ThreadPoolExecutor for parallel CPU-bound language detection
    start_time = time.time()
    print(f"[Core {core_id_log}] Starting parallel gcld3 processing on {len(df)} samples with {NUM_CPU_WORKERS} workers...", flush=True)
    
    try:
        # Apply the language check function to all 'text' rows in parallel
        with ThreadPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor: 
            is_english_results = list(executor.map(check_language, df['text'].tolist()))
        
        elapsed = time.time() - start_time
        english_count = sum(is_english_results)
        print(f"[Core {core_id_log}] ✅ gcld3 processing complete in {elapsed:.2f} seconds. Found {english_count} English samples.", flush=True)
        
        # 2. Add the new column
        df['is_english_reliable'] = is_english_results

        # 3. Overwrite the local file with the new, processed version
        df.to_parquet(local_path, index=False)
        print(f"[Core {core_id_log}] ✅ Saved processed shard over local file. Filtering is now pre-computed.", flush=True)
        
    except Exception as e:
        # Log the error, but since the core cannot proceed without filtering, we should return None.
        print(f"[Core {core_id_log}] ❌ CRITICAL Error during gcld3 processing: {e}. Returning None.", flush=True)
        return None

    # 4. Return the local path to the PROCESSED file
    return local_path


# --- Checkpoint Access Function (Existing) ---

def get_checkpoint_start_index(core_id, local_dir="/tmp/checkpoints"):
    """
    Reads the last 'samples_seen' index from the core's checkpoint file 
    in GCS.

    Args:
        core_id (int): The ID of the current TPU core.
        local_dir (str): A temporary local directory for downloading the checkpoint.

    Returns:
        int/None: The samples seen index if found and valid, otherwise None.
    """
    core_id_log = get_core_ordinal()
    gcs_blob_path = os.path.join(GCS_CHECKPOINT_PREFIX, f"core_{core_id}", CHECKPOINT_FILENAME)
    # The local directory structure ensures the checkpoint file is isolated from data shards
    local_ckpt_path = os.path.join(local_dir, f"core_{core_id}_", CHECKPOINT_FILENAME) 

    print(f"[Core {core_id_log}] Reading checkpoint index from: gs://{BUCKET_NAME}/{gcs_blob_path}", flush=True)

    # Use the shared download function
    if not download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path):
        print(f"[Core {core_id_log}] ❌ CHECKPOINT READ FAILED: file not found or failed to download. Starting from index 0.", flush=True)
        return 0

    try:
        with open(local_ckpt_path, "r") as f:
            progress = json.load(f)
            index = progress.get("samples_seen")
            if index is not None and isinstance(index, int) and index >= 0:
                print(f"[Core {core_id_log}] ✅ Checkpoint read successful. Starting row index: {index}", flush=True)
                return index
            else:
                # If file exists but content is invalid, treat as starting from 0
                print(f"[Core {core_id_log}] ❌ 'samples_seen' missing or invalid in checkpoint. Starting from index 0.", flush=True)
                return 0
    except Exception as e:
        # If file exists but is corrupted/unreadable JSON, treat as starting from 0
        print(f"[Core {core_id_log}] ❌ Error parsing checkpoint JSON: {e}. Starting from index 0.", flush=True)
        return 0
    finally:
        # Clean up the local checkpoint file after reading
        if os.path.exists(local_ckpt_path):
            os.remove(local_ckpt_path)