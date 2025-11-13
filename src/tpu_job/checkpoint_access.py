import os
import json
import time
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

# --- Configuration (These can be imported and overridden if needed) ---
DEFAULT_BUCKET_NAME = "encoder-models-2"
DEFAULT_FILENAME = "progress.json"
DEFAULT_GCS_PREFIX = "tensorcore-checkpoints-v2-init"

# --- 1. GCS Helper Functions ---

def _get_gcs_client_and_blob(bucket_name, gcs_blob_path):
    """Initializes GCS client, bucket, and blob object."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    return blob

def upload_file_to_gcs(local_path, gcs_blob_path, bucket_name=DEFAULT_BUCKET_NAME):
    """Uploads a local file to GCS."""
    core_id = get_core_ordinal()
    try:
        blob = _get_gcs_client_and_blob(bucket_name, gcs_blob_path)
        blob.upload_from_filename(local_path)
        print(f"[Core {core_id}] ✅ Upload SUCCESS: {local_path} -> gs://{bucket_name}/{gcs_blob_path}", flush=True)
        return True
    except Exception as e:
        print(f"[Core {core_id}] ❌ GCS UPLOAD FAILED for {local_path}. Error: {e}", flush=True)
        return False

def download_file_from_gcs(local_path, gcs_blob_path, bucket_name=DEFAULT_BUCKET_NAME):
    """Downloads a file from GCS. Creates parent directories if needed."""
    core_id = get_core_ordinal()
    try:
        blob = _get_gcs_client_and_blob(bucket_name, gcs_blob_path)
        
        if not blob.exists():
            return False 
            
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"[Core {core_id}] ❌ GCS DOWNLOAD FAILED for {gcs_blob_path}. Error: {e}", flush=True)
        return False

# --- 2. Checkpoint Management Functions ---

def get_checkpoint_gcs_path(core_id, gcs_prefix=DEFAULT_GCS_PREFIX, filename=DEFAULT_FILENAME):
    """Constructs the full GCS path for a given core's checkpoint file."""
    # Example path: tensorcore-checkpoints-v2-init/core_3/progress.json
    gcs_blob_base_dir = f"{gcs_prefix}/core_{core_id}"
    return os.path.join(gcs_blob_base_dir, filename)


def load_checkpoint(core_id, local_dir="/tmp/checkpoints", gcs_prefix=DEFAULT_GCS_PREFIX, filename=DEFAULT_FILENAME):
    """
    Downloads the checkpoint file from GCS and loads its content.

    Args:
        core_id (int): The ID of the core whose checkpoint to load.
        local_dir (str): Temporary local directory to store the downloaded file.
        ... other config args
        
    Returns:
        dict: The checkpoint data, or None if download/loading failed.
    """
    gcs_path = get_checkpoint_gcs_path(core_id, gcs_prefix, filename)
    local_path = os.path.join(local_dir, f"core_{core_id}_{filename}")
    core_id_log = get_core_ordinal()

    if not download_file_from_gcs(local_path, gcs_path):
        print(f"[Core {core_id_log}] Checkpoint file not found on GCS for Core {core_id} at {gcs_path}. Returning empty checkpoint.", flush=True)
        # Create a default structure if not found, to be consistent
        return {"core_id": core_id, "samples_seen": 0}

    try:
        with open(local_path, "r") as f:
            checkpoint_data = json.load(f)
            print(f"[Core {core_id_log}] Checkpoint loaded successfully for Core {core_id}.", flush=True)
            return checkpoint_data
    except Exception as e:
        print(f"[Core {core_id_log}] ❌ Error loading JSON from {local_path}: {e}", flush=True)
        return None
    finally:
        # Clean up local file after reading
        if os.path.exists(local_path):
            os.remove(local_path)


def save_checkpoint(core_id, checkpoint_data, local_dir="/tmp/checkpoints", gcs_prefix=DEFAULT_GCS_PREFIX, filename=DEFAULT_FILENAME):
    """
    Saves the checkpoint data locally and uploads it to GCS.

    Args:
        core_id (int): The ID of the core whose checkpoint to save.
        checkpoint_data (dict): The data structure to save (must be serializable).
        local_dir (str): Temporary local directory to store the file before upload.
        ... other config args

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    gcs_path = get_checkpoint_gcs_path(core_id, gcs_prefix, filename)
    local_path = os.path.join(local_dir, f"core_{core_id}_{filename}")
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        # 1. Write locally
        with open(local_path, "w") as f:
            json.dump(checkpoint_data, f, indent=4)
            
        # 2. Upload to GCS
        upload_success = upload_file_to_gcs(local_path=local_path, gcs_blob_path=gcs_path)
        return upload_success
    except Exception as e:
        print(f"[Core {get_core_ordinal()}] ❌ Failed to save checkpoint for Core {core_id}. Error: {e}", flush=True)
        return False
    finally:
        # Clean up local file after uploading
        if os.path.exists(local_path):
            os.remove(local_path)


def reset_checkpoint_samples(core_id, local_dir="/tmp/checkpoints", gcs_prefix=DEFAULT_GCS_PREFIX, filename=DEFAULT_FILENAME):
    """
    NEW FUNCTION: Downloads a checkpoint, sets 'samples_seen' to 0, 
    updates the last reset time, and uploads the modified checkpoint.
    """
    core_id_log = get_core_ordinal()
    print(f"[Core {core_id_log}] Attempting to RESET samples_seen for Checkpoint {core_id}...", flush=True)

    # 1. Load existing checkpoint (or get default if it doesn't exist)
    progress = load_checkpoint(core_id, local_dir, gcs_prefix, filename)

    if progress is None:
        print(f"[Core {core_id_log}] ❌ Reset FAILED: Could not load or initialize checkpoint data for Core {core_id}.", flush=True)
        return False

    if isinstance(progress, int):
        original_samples = progress
    else:
        original_samples = progress.get("samples_seen", "N/A")
    
    # 2. Modify / FORCE RESET
    progress["samples_seen"] = 0
    progress["last_reset_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # 3. Save and upload the modified checkpoint
    save_success = save_checkpoint(core_id, progress, local_dir, gcs_prefix, filename)

    if save_success:
        print(f"[Core {core_id_log}] ✅ RESET SUCCESS for Core {core_id}. Samples forced from {original_samples} to 0.", flush=True)
    else:
        print(f"[Core {core_id_log}] ❌ Reset FAILED for Core {core_id}.", flush=True)

    return save_success