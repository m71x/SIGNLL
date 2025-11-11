import os
import json
import time
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# --- Configuration ---
# NOTE: All GCS paths used here are expected to exist and contain a progress.json file.
BUCKET_NAME = "encoder-models-2"
CHECKPOINT_FILENAME = "progress.json"

# Phase 1: Test path for a single core's RMW sequence
GCS_PREFIX_PHASE_1 = "tensorcore-checkpoints" 

# Phase 2: Test path for sharded, simultaneous RMW across 32 existing folders
GCS_PREFIX_SHARDED_RMW = "tensorcore-checkpoints-v2-sharded" 
TOTAL_TARGET_FOLDERS = 32 # The number of existing folders to test

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
            print(f"[Core {core_id}] Warning: Blob not found at GCS path {gcs_blob_path}. Skipping download.", flush=True)
            return False 
            
        # Ensure the local directory exists for the temporary file
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"[Core {core_id}] !!! GCS DOWNLOAD FAILED for {gcs_blob_path} !!! Error: {e}", flush=True)
        return False

# --- General Purpose Checkpoint Access and Modify Function (RMW) ---
def access_and_modify_checkpoint(core_id, local_dir, gcs_prefix, target_core_id, samples_delta):
    """
    Performs a Read-Modify-Write (RMW) cycle on a core-specific checkpoint file 
    presumed to exist in GCS.
    """
    
    gcs_blob_base_dir = f"{gcs_prefix}/core_{target_core_id}"
    local_ckpt_filename = f"{gcs_prefix}_core_{target_core_id}_{CHECKPOINT_FILENAME}"
    local_ckpt_path = os.path.join(local_dir, local_ckpt_filename)
    gcs_blob_path = os.path.join(gcs_blob_base_dir, CHECKPOINT_FILENAME)
    
    # 1. READ: Attempt to download existing checkpoint
    download_success = download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path)

    # 2. LOAD/HANDLE NON-EXISTENCE: Load or fail if file not found
    progress = {}
    
    if not download_success:
        # Since we are testing RMW on *existing* data, treat non-existence as a failure
        # that must be handled by the caller, or throw an error.
        raise FileNotFoundError(f"Checkpoint file expected but not found at GCS path: {gcs_blob_path}")

    try:
        with open(local_ckpt_path, "r") as f:
            progress = json.load(f)
    except Exception as e:
        # File found but corrupted
        print(f"[Core {core_id}] !!! Checkpoint file corrupted at {gcs_blob_path} !!! Error: {e}", flush=True)
        os.remove(local_ckpt_path)
        raise RuntimeError(f"Corrupted checkpoint found for Core {target_core_id}.")

    # Safety check for required key
    if "samples_seen" not in progress:
        print(f"[Core {core_id}] !!! Checkpoint missing 'samples_seen' key at {gcs_blob_path} !!! Initializing it to 0.", flush=True)
        progress["samples_seen"] = 0
        
    # 3. MODIFY: Update the checkpoint state
    original_samples = progress["samples_seen"]
    
    if samples_delta != 0:
        progress["samples_seen"] += samples_delta
        print(f"[Core {core_id}] RMW: {gcs_prefix}/core_{target_core_id} modified: {original_samples} + {samples_delta} = {progress['samples_seen']}", flush=True)
    
    progress["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    progress["modifying_core"] = core_id

    # 4. WRITE & UPLOAD: Save locally and upload (overwriting the existing blob)
    with open(local_ckpt_path, "w") as f:
        json.dump(progress, f, indent=4)
    
    upload_success = upload_file_to_gcs(
        local_path=local_ckpt_path,
        bucket_name=BUCKET_NAME,
        gcs_blob_path=gcs_blob_path
    )
    
    # 5. Clean up local file and report
    os.remove(local_ckpt_path)

    if not upload_success:
        raise RuntimeError(f"Failed checkpoint RMW cycle for Core {core_id} at GCS path {gcs_blob_path}.")

    return progress["samples_seen"]


# --- Phase 1: General RMW Test (Read-Add-Subtract) on a single path ---
def run_phase_1_rmw_test(core_id, local_dir):
    """Performs a sequence of RMW operations on the core's designated Phase 1 checkpoint."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 1: General RMW Test (Expected existing file)\n{'='*50}", flush=True)
    
    try:
        # Use core_id as the target_core_id for this single path test
        target_core_id = core_id 

        # --- Step 1: Read Initial Samples (Delta 0) ---
        print(f"[Core {core_id}] Step 1: Reading initial samples_seen (Delta 0)")
        initial_samples = access_and_modify_checkpoint(core_id, local_dir, GCS_PREFIX_PHASE_1, target_core_id, samples_delta=0)
        print(f"[Core {core_id}] Result 1: Samples seen is currently {initial_samples}")

        # --- Step 2: Add 100 Samples (Delta +100) ---
        print(f"\n[Core {core_id}] Step 2: Adding 100 samples (Delta +100)")
        new_samples = access_and_modify_checkpoint(core_id, local_dir, GCS_PREFIX_PHASE_1, target_core_id, samples_delta=100)
        print(f"[Core {core_id}] Result 2: Samples seen after adding 100 is {new_samples}")

        # --- Step 3: Subtract 100 Samples (Delta -100) ---
        print(f"\n[Core {core_id}] Step 3: Subtracting 100 samples (Delta -100)")
        final_samples = access_and_modify_checkpoint(core_id, local_dir, GCS_PREFIX_PHASE_1, target_core_id, samples_delta=-100)
        print(f"[Core {core_id}] Result 3: Samples seen after subtracting 100 is {final_samples}")

        print(f"\n[Core {core_id}] ✅ Phase 1 Test Sequence Complete.")
        
    except FileNotFoundError:
         print(f"[Core {core_id}] ❌ Phase 1 FAILED: Checkpoint was not found. Please ensure {GCS_PREFIX_PHASE_1}/core_{core_id}/{CHECKPOINT_FILENAME} exists.", flush=True)
    except RuntimeError as e:
        print(f"[Core {core_id}] ❌ Phase 1 FAILED: {e}", flush=True)


# --- Phase 2: Sharded RMW Test on 32 Existing Checkpoints ---
def run_sharded_rmw_test(core_id, total_cores, local_dir):
    """Performs an RMW cycle on a shard of 32 existing GCS checkpoints."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 2: Sharded RMW Test on {TOTAL_TARGET_FOLDERS} Existing Checkpoints\n{'='*50}", flush=True)
    
    # 1. Determine which folders this core is responsible for
    folders_assigned = []
    for i in range(TOTAL_TARGET_FOLDERS):
        # Shard the work: folder 'i' is handled by core 'i % total_cores'
        if i % total_cores == core_id:
            folders_assigned.append(i)
    
    print(f"[Core {core_id}] Responsible for RMW on existing folders: {folders_assigned}", flush=True)

    rmw_passed_count = 0
    rmw_failed_count = 0
    
    for folder_index in folders_assigned:
        try:
            # RMW cycle: Add a unique delta (e.g., 10 * core_id) to the 'samples_seen'
            delta = core_id + 1 
            access_and_modify_checkpoint(
                core_id=core_id, 
                local_dir=local_dir, 
                gcs_prefix=GCS_PREFIX_SHARDED_RMW, 
                target_core_id=folder_index, 
                samples_delta=delta
            )
            print(f"[Core {core_id}] ✅ RMW PASS for Folder {folder_index}. Added delta {delta}.", flush=True)
            rmw_passed_count += 1
            
        except FileNotFoundError:
             print(f"[Core {core_id}] ❌ RMW FAILED for Folder {folder_index}: File not found. Please ensure {GCS_PREFIX_SHARDED_RMW}/core_{folder_index}/{CHECKPOINT_FILENAME} exists.", flush=True)
             rmw_failed_count += 1
        except RuntimeError as e:
            print(f"[Core {core_id}] ❌ RMW FAILED for Folder {folder_index}: Operation failed. Error: {e}", flush=True)
            rmw_failed_count += 1

    # 4. Report results
    print(f"\n[Core {core_id}] PHASE 2 RESULTS: RMW Passed {rmw_passed_count}/{len(folders_assigned)}. Failed {rmw_failed_count}.", flush=True)


# --- Worker function for each TPU core ---
def _mp_fn(index):
    core_id = xm.get_ordinal()
    total_cores = xm.xrt_world_size()
    
    # Create a unique local temporary directory for this core
    local_dir = f"/tmp/tpu_ckpt_test_{core_id}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[Core {core_id}] Using device {xm.torch_xla.device()} / Total TPU cores: {total_cores}", flush=True)

    # --- 1. PHASE 1: READ-MODIFY-WRITE TEST ---
    run_phase_1_rmw_test(core_id, local_dir)
    
    # Synchronize all cores before starting the second phase
    xm.rendezvous("phase_1_complete")

    # --- 2. PHASE 2: SHARDED RMW TEST ---
    run_sharded_rmw_test(core_id, total_cores, local_dir)
    
    # Final Synchronization
    xm.rendezvous("phase_2_complete")
    
    print(f"[Core {core_id}] ALL PHASES COMPLETE.", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    # This must be run with the appropriate launcher for PyTorch XLA on TPU
    xmp.spawn(_mp_fn, args=())
