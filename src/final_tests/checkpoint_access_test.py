import os
import json
import time
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# --- Configuration ---
BUCKET_NAME = "encoder-models-2"
CHECKPOINT_FILENAME = "progress.json"

# Phase 1: Existing checkpoint path (Read/Modify/Write test)
GCS_PREFIX_PHASE_1 = "tensorcore-checkpoints" 

# Phase 2: New initialization path (New 32 files) - ensures no conflict
GCS_PREFIX_PHASE_2 = "tensorcore-checkpoints-v2-init" 
TOTAL_TARGET_FOLDERS = 32 # The number of new folders to create

# --- Helper: GCS Client Setup ---
def get_gcs_client_and_blob(bucket_name, gcs_blob_path):
    """Initializes GCS client, bucket, and blob object."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    return blob

# --- Helper: Upload file to GCS ---
def upload_file_to_gcs(local_path, bucket_name, gcs_blob_path):
    """Uploads a file to GCS using the official client library."""
    core_id = xm.get_ordinal()
    try:
        blob = get_gcs_client_and_blob(bucket_name, gcs_blob_path)
        blob.upload_from_filename(local_path)
        return True
    except Exception as e:
        print(f"[Core {core_id}] !!! GCS UPLOAD FAILED for {local_path} !!! Error: {e}", flush=True)
        return False

# --- Helper: Download file from GCS ---
def download_file_from_gcs(local_path, bucket_name, gcs_blob_path):
    """Downloads a file from GCS using the official client library."""
    core_id = xm.get_ordinal()
    try:
        blob = get_gcs_client_and_blob(bucket_name, gcs_blob_path)
        
        if not blob.exists():
            return False 
            
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"[Core {core_id}] !!! GCS DOWNLOAD FAILED for {gcs_blob_path} !!! Error: {e}", flush=True)
        return False

# --- Phase 1: Read-Modify-Write Test ---
def run_read_modify_write_test(core_id, local_dir):
    """Reads the existing progress file, modifies it, and uploads it."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 1: Read-Modify-Write Test (Existing Checkpoint)\n{'='*50}", flush=True)
    
    # 1. Define Paths for Phase 1
    gcs_blob_base_dir = f"{GCS_PREFIX_PHASE_1}/core_{core_id}"
    local_ckpt_path = os.path.join(local_dir, f"{GCS_PREFIX_PHASE_1}_{CHECKPOINT_FILENAME}")
    gcs_blob_path = os.path.join(gcs_blob_base_dir, CHECKPOINT_FILENAME)
    
    SAMPLES_TO_ADD = 100
    
    # 2. READ: Attempt to download existing checkpoint
    download_success = download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path)

    # 3. INITIALIZE/LOAD: Load or create the progress data
    if download_success:
        try:
            with open(local_ckpt_path, "r") as f:
                progress = json.load(f)
            print(f"[Core {core_id}] Loaded existing checkpoint. Samples: {progress['samples_seen']}", flush=True)
        except Exception:
            print(f"[Core {core_id}] Checkpoint file corrupted or failed to load. Creating new.", flush=True)
            progress = {"core_id": core_id, "samples_seen": 0}
    else:
        print(f"[Core {core_id}] Checkpoint not found on GCS. Creating initial file with samples_seen: 0.", flush=True)
        progress = {"core_id": core_id, "samples_seen": 0}

    # 4. MODIFY: Update the checkpoint state
    original_samples = progress["samples_seen"]
    progress["samples_seen"] += SAMPLES_TO_ADD
    progress["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    
    print(f"[Core {core_id}] MODIFIED: {original_samples} + {SAMPLES_TO_ADD} = {progress['samples_seen']}", flush=True)

    # 5. WRITE & UPLOAD: Save locally and upload
    with open(local_ckpt_path, "w") as f:
        json.dump(progress, f, indent=4)
    
    upload_success = upload_file_to_gcs(
        local_path=local_ckpt_path,
        bucket_name=BUCKET_NAME,
        gcs_blob_path=gcs_blob_path
    )

    # 6. Final Verification
    if upload_success:
        print(f"[Core {core_id}] ✅ Phase 1 Success! Uploaded new samples_seen: {progress['samples_seen']} to gs://{BUCKET_NAME}/{gcs_blob_path}", flush=True)
    else:
        raise RuntimeError(f"Failed Phase 1 read-modify-write cycle for Core {core_id}.")
        
    # Clean up local file
    os.remove(local_ckpt_path)


# --- Phase 2: Create 32 New Checkpoints (Sharded) ---
def initialize_32_checkpoints(core_id, total_cores, local_dir):
    """Collaboratively creates 32 new checkpoint files on GCS."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 2: Initializing {TOTAL_TARGET_FOLDERS} New Checkpoints\n{'='*50}", flush=True)
    
    # 1. Determine which folders this core is responsible for
    folders_assigned = []
    for i in range(TOTAL_TARGET_FOLDERS):
        # Shard the work: folder 'i' is handled by core 'i % total_cores'
        if i % total_cores == core_id:
            folders_assigned.append(i)
    
    print(f"[Core {core_id}] Responsible for initializing folders: {folders_assigned}", flush=True)
    
    if not folders_assigned:
        print(f"[Core {core_id}] No folders assigned for initialization. Skipping uploads.", flush=True)
        return

    # 2. Loop through assigned folders, create file, and upload
    upload_count = 0
    for folder_index in folders_assigned:
        # Define unique paths for the new checkpoint
        gcs_blob_base_dir = f"{GCS_PREFIX_PHASE_2}/core_{folder_index}"
        local_ckpt_path = os.path.join(local_dir, f"{GCS_PREFIX_PHASE_2}_core_{folder_index}_{CHECKPOINT_FILENAME}")
        gcs_blob_path = os.path.join(gcs_blob_base_dir, CHECKPOINT_FILENAME)
        
        # Create initial progress data (samples_seen=0)
        progress = {"core_id": folder_index, "samples_seen": 0, "initialization_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        
        # Write locally
        with open(local_ckpt_path, "w") as f:
            json.dump(progress, f, indent=4)
            
        # Upload to GCS
        upload_success = upload_file_to_gcs(
            local_path=local_ckpt_path,
            bucket_name=BUCKET_NAME,
            gcs_blob_path=gcs_blob_path
        )
        
        if upload_success:
            upload_count += 1
            print(f"[Core {core_id}] Initialized Folder {folder_index}/{TOTAL_TARGET_FOLDERS} successfully.", flush=True)
        else:
            print(f"[Core {core_id}] FAILED to initialize Folder {folder_index}!", flush=True)
            
        # Clean up local file
        os.remove(local_ckpt_path)

    print(f"[Core {core_id}] ✅ Phase 2 Success! Created {upload_count} of {len(folders_assigned)} assigned checkpoints.", flush=True)


# --- Worker function for each TPU core ---
def _mp_fn(index):
    core_id = xm.get_ordinal()
    total_cores = xm.xrt_world_size()
    
    # Create a unique local temporary directory for this core
    local_dir = f"/tmp/tpu_ckpt_test_{core_id}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[Core {core_id}] Using device {xm.torch_xla.device()} / Total TPU cores: {total_cores}", flush=True)

    # --- 1. PHASE 1: READ-MODIFY-WRITE TEST ---
    run_read_modify_write_test(core_id, local_dir)
    
    # Synchronize all cores before starting the second phase
    xm.rendezvous("phase_1_complete")

    # --- 2. PHASE 2: CREATE 32 NEW CHECKPOINTS ---
    initialize_32_checkpoints(core_id, total_cores, local_dir)
    
    # Final Synchronization
    xm.rendezvous("phase_2_complete")
    
    # Clean up the core's temporary directory
    # Note: On a shared filesystem, this might be tricky, but for /tmp on the worker, it's fine.
    # os.rmdir(local_dir)
    
    print(f"[Core {core_id}] ALL PHASES COMPLETE.", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    # Ensure all necessary imports are available and authenticated for GCS access
    # This must be run with the appropriate launcher for PyTorch XLA on TPU, e.g.,
    # python -m torch_xla.distributed.xla_multiprocessing --workers=all tpu_checkpoint_sequencing_test.py
    xmp.spawn(_mp_fn, args=())