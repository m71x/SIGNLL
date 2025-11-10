import os
import json
# Removed 'subprocess'
# Added 'google.cloud' for the official GCS client
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

#MAKE SURE TO USE GCLOUD CLIENT CALL, NOT GSUTIL TO UPLOAD
#ALSO RUN WITH --workers = all FLAG
# --- Configuration ---
BUCKET_NAME = "encoder-models"
GCS_PREFIX = "tensorcore-checkpoints"

# --- Function: Upload file using GCS Client ---
def upload_file_to_gcs(local_path, bucket_name, gcs_blob_path):
    """Uploads a file to GCS using the official client library."""
    try:
        # Initialize the GCS client (must be authenticated)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_path)
        
        # Perform the upload
        blob.upload_from_filename(local_path)
        return True
    except Exception as e:
        # Print the error for debugging, but don't crash the whole process yet
        print(f"!!! GCS UPLOAD FAILED for {local_path} !!! Error: {e}", flush=True)
        return False


# --- Worker function for each TPU core ---
def _mp_fn(index):
    # Identify TPU device and rank info
    device = xm.torch_xla.device()  # Using the recommended alias
    core_id = xm.get_ordinal()
    total_cores = xm.xrt_world_size()

    print(f"[Core {core_id}] Using device {device} / Total TPU cores: {total_cores}", flush=True)

    # Construct GCS and local checkpoint paths
    # Note: The GCS client uses a blob path, not the 'gs://' URL prefix
    GCS_BUCKET_NAME = BUCKET_NAME
    gcs_blob_base_dir = f"{GCS_PREFIX}/core_{core_id}"

    local_dir = f"/tmp/core_{core_id}"
    os.makedirs(local_dir, exist_ok=True)

    local_ckpt_path = os.path.join(local_dir, "progress.json")
    gcs_blob_path = os.path.join(gcs_blob_base_dir, "progress.json")


    # Create checkpoint file
    progress = {"core_id": core_id, "samples_seen": 0}
    with open(local_ckpt_path, "w") as f:
        json.dump(progress, f)

    print(f"[Core {core_id}] Created local checkpoint: {local_ckpt_path}", flush=True)

    # --- Replaced subprocess.run with GCS Client call ---
    success = upload_file_to_gcs(
        local_path=local_ckpt_path,
        bucket_name=GCS_BUCKET_NAME,
        gcs_blob_path=gcs_blob_path
    )

    if success:
        print(f"[Core {core_id}] Uploaded checkpoint to gs://{GCS_BUCKET_NAME}/{gcs_blob_path} ✅", flush=True)
    else:
        # If upload fails, you may want to raise an error to stop the run
        raise RuntimeError(f"Failed to upload checkpoint for Core {core_id}. See previous error message.")

    # Synchronize across all TPU cores
    xm.rendezvous("checkpoint_sync")

    print(f"[Core {core_id}] Initialization complete.", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())