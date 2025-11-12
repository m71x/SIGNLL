import os
import json
from google.cloud import storage

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

def download_data_shard(core_id, local_shard_path):
    """
    Downloads the data shard corresponding to the core ID from GCS to 
    the specified local path.

    Args:
        core_id (int): The ID of the current TPU core.
        local_shard_path (str): The determined local path to save the shard.

    Returns:
        bool: True if the shard was downloaded successfully, False otherwise.
    """
    core_id_log = get_core_ordinal() 

    # 1. Determine GCS path
    # Parquet filename format uses the core_id directly as the shard_index
    filename = PARQUET_FILE_FORMAT.format(shard_index=core_id)
    gcs_blob_path = os.path.join(GCS_DATA_PREFIX, filename)
    
    print(f"[Core {core_id_log}] Attempting to download data shard {filename} to {local_shard_path}...", flush=True)

    # 2. Use existing GCS helper function
    return download_file_from_gcs(local_shard_path, BUCKET_NAME, gcs_blob_path)


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