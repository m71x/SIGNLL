import os
import json
import time
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# --- Configuration ---
BUCKET_NAME = "encoder-models"
CHECKPOINT_FILENAME = "progress.json"

# Phases 1, 2: Reset, and verification path
GCS_PREFIX = "tensorcore-checkpoints-v2-init" 
TOTAL_TARGET_FOLDERS = 32 # The number of existing folders

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


# --- Phase 1: Reset All 32 Checkpoints to Zero ---
def run_phase_1_reset_all_checkpoints(core_id, total_cores, local_dir):
    """Collaboratively resets all 32 existing checkpoint files to samples_seen: 0."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 1: Resetting All {TOTAL_TARGET_FOLDERS} Existing Checkpoints to ZERO\n{'='*50}", flush=True)
    
    # 1. Determine which folders this core is responsible for
    folders_assigned = []
    for i in range(TOTAL_TARGET_FOLDERS):
        if i % total_cores == core_id:
            folders_assigned.append(i)
    
    print(f"[Core {core_id}] Responsible for resetting folders: {folders_assigned}", flush=True)

    success_count = 0
    for folder_index in folders_assigned:
        # Define unique paths
        gcs_blob_base_dir = f"{GCS_PREFIX}/core_{folder_index}"
        local_ckpt_path = os.path.join(local_dir, f"reset_core_{folder_index}_{CHECKPOINT_FILENAME}")
        gcs_blob_path = os.path.join(gcs_blob_base_dir, CHECKPOINT_FILENAME)
        
        # 1. Download existing file (to preserve other metadata)
        # This function returns False if the file doesn't exist, which is a potential issue if the files were not created previously.
        download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path)

        # 2. Load or default to initial structure
        progress = {"core_id": folder_index, "samples_seen": 0} 
        file_exists = os.path.exists(local_ckpt_path)
        
        if file_exists:
            try:
                with open(local_ckpt_path, "r") as f:
                    progress = json.load(f)
            except:
                print(f"[Core {core_id}] Warning: Could not load JSON for Folder {folder_index}. Using default reset state.", flush=True)
                pass 
        else:
            print(f"[Core {core_id}] Warning: Checkpoint file not found for Folder {folder_index}. Creating new file with reset state.", flush=True)
        
        # 3. FORCE RESET
        original_samples = progress.get("samples_seen", "N/A")
        progress["samples_seen"] = 0
        progress["last_reset_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        # 4. WRITE & UPLOAD
        with open(local_ckpt_path, "w") as f:
            json.dump(progress, f, indent=4)
            
        upload_success = upload_file_to_gcs(local_path=local_ckpt_path, bucket_name=BUCKET_NAME, gcs_blob_path=gcs_blob_path)
        
        # 5. Clean up
        if os.path.exists(local_ckpt_path):
             os.remove(local_ckpt_path)
        
        if upload_success:
            print(f"[Core {core_id}] ✅ RESET SUCCESS for Folder {folder_index}. Samples forced from {original_samples} to 0.", flush=True)
            success_count += 1
        else:
            print(f"[Core {core_id}] ❌ RESET FAILED for Folder {folder_index}.", flush=True)

    print(f"\n[Core {core_id}] PHASE 1 RESULTS: Successfully reset {success_count}/{len(folders_assigned)} checkpoints.", flush=True)


# --- Phase 2: Final Verification of Reset State ---
def run_phase_2_final_verification(core_id, total_cores, local_dir):
    """Verifies that the 32 files now consistently contain 'samples_seen': 0."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 2: Final Verification (Expected samples_seen: 0)\n{'='*50}", flush=True)
    
    # 1. Determine which folders this core is responsible for
    folders_assigned = []
    for i in range(TOTAL_TARGET_FOLDERS):
        if i % total_cores == core_id:
            folders_assigned.append(i)
    
    print(f"[Core {core_id}] Responsible for verifying folders: {folders_assigned}", flush=True)

    test_passed_count = 0
    test_failed_count = 0
    EXPECTED_SAMPLES = 0
    
    for folder_index in folders_assigned:
        # Define paths for the checkpoint location
        gcs_blob_base_dir = f"{GCS_PREFIX}/core_{folder_index}"
        local_ckpt_path = os.path.join(local_dir, f"verification_core_{folder_index}_{CHECKPOINT_FILENAME}")
        gcs_blob_path = os.path.join(gcs_blob_base_dir, CHECKPOINT_FILENAME)
        
        # 2. Attempt to download
        download_success = download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path)

        if not download_success:
            print(f"[Core {core_id}] ❌ Verification FAILED for Folder {folder_index}: File not found or download failed. CHECKPOINT MUST EXIST.", flush=True)
            test_failed_count += 1
            continue

        # 3. Verify content
        try:
            with open(local_ckpt_path, "r") as f:
                progress = json.load(f)
            
            # Check for core_id consistency and samples_seen value
            actual_samples = progress.get("samples_seen")
            if progress.get("core_id") == folder_index and actual_samples == EXPECTED_SAMPLES:
                print(f"[Core {core_id}] ✅ Verification PASS for Folder {folder_index}. Content confirmed (samples_seen: {EXPECTED_SAMPLES}).", flush=True)
                test_passed_count += 1
            else:
                print(f"[Core {core_id}] ❌ Verification FAILED for Folder {folder_index}: Content mismatch. Expected Samples: {EXPECTED_SAMPLES}, Got: {actual_samples}", flush=True)
                test_failed_count += 1
                
        except Exception as e:
            print(f"[Core {core_id}] ❌ Verification FAILED for Folder {folder_index}: Error reading content: {e}", flush=True)
            test_failed_count += 1
            
        # Clean up local file
        os.remove(local_ckpt_path)

    # 4. Report results
    print(f"\n[Core {core_id}] PHASE 2 RESULTS: Passed {test_passed_count}/{len(folders_assigned)}. Failed {test_failed_count}.", flush=True)


# --- Worker function for each TPU core ---
def _mp_fn(index):
    core_id = xm.get_ordinal()
    total_cores = xm.xrt_world_size()
    
    # Create a unique local temporary directory for this core
    local_dir = f"/tmp/tpu_ckpt_test_{core_id}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[Core {core_id}] Using device {xm.torch_xla.device()} / Total TPU cores: {total_cores}", flush=True)

    # --- 1. PHASE 1: RESET ALL CHECKPOINTS TO ZERO ---
    run_phase_1_reset_all_checkpoints(core_id, total_cores, local_dir)
    
    # Synchronize all cores before starting the final verification
    xm.rendezvous("phase_1_complete")

    # --- 2. PHASE 2: FINAL VERIFICATION OF RESET STATE ---
    run_phase_2_final_verification(core_id, total_cores, local_dir)
    
    # Final Synchronization
    xm.rendezvous("phase_2_complete")
    
    # Clean up the core's temporary directory
    # os.rmdir(local_dir)
    
    print(f"[Core {core_id}] ALL PHASES COMPLETE.", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    # This must be run with the appropriate launcher for PyTorch XLA on TPU, e.g.,
    # python -m torch_xla.distributed.xla_multiprocessing --workers=all tpu_checkpoint_sequencing_test.py
    xmp.spawn(_mp_fn, args=())