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
        bucket_name (str): The name of the GCS bucket.
        gcs_blob_path (str): The path to the file in the GCS bucket.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    core_id = get_core_ordinal()
    try:
        blob = _get_gcs_client_and_blob(bucket_name, gcs_blob_path)

        if not blob.exists():
            print(f"[Core {core_id}] ⚠️ GCS file not found: {gcs_blob_path}", flush=True)
            return False

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"[Core {core_id}] ✅ Downloaded: {gcs_blob_path} to {local_path}", flush=True)
        return True
    except Exception as e:
        print(f"[Core {core_id}] !!! GCS DOWNLOAD FAILED for {gcs_blob_path} !!! Error: {e}", flush=True)
        return False


# --- Helper: Checkpoint Reading ---

def read_checkpoint_index(core_id, local_dir):
    """
    Downloads the checkpoint file specific to the core_id from GCS and 
    extracts the 'samples_seen' integer value.
    
    Args:
        core_id (int): The ID of the current TPU core.
        local_dir (str): A temporary local directory for downloading the checkpoint.

    Returns:
        int/None: The samples seen index if found and valid, otherwise None.
    """
    gcs_blob_path = os.path.join(GCS_CHECKPOINT_PREFIX, f"core_{core_id}", CHECKPOINT_FILENAME)
    local_ckpt_path = os.path.join(local_dir, CHECKPOINT_FILENAME)

    print(f"[Core {core_id}] Reading checkpoint index from: gs://{BUCKET_NAME}/{gcs_blob_path}", flush=True)

    # Use the shared download function
    if not download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path):
        print(f"[Core {core_id}] ❌ CHECKPOINT READ FAILED: file not found or failed to download.", flush=True)
        return None

    try:
        with open(local_ckpt_path, "r") as f:
            progress = json.load(f)
            index = progress.get("samples_seen")
            if index is not None and isinstance(index, int):
                print(f"[Core {core_id}] ✅ Checkpoint read successful. Starting row index: {index}", flush=True)
                return index
            else:
                print(f"[Core {core_id}] ❌ 'samples_seen' missing or invalid.", flush=True)
                return None
    except Exception as e:
        print(f"[Core {core_id}] ❌ Error parsing checkpoint JSON: {e}", flush=True)
        return None
    finally:
        # Clean up local checkpoint file
        if os.path.exists(local_ckpt_path):
            os.remove(local_ckpt_path)