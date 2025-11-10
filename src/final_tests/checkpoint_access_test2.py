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

# Phase 2 & 3: New initialization and verification path
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

# --- General Purpose Checkpoint Access and Modify Function ---
def access_and_modify_checkpoint(core_id: int, local_dir: str, gcs_prefix: str, samples_delta: int) -> int:
    """
    Downloads a core-specific progress file, modifies the samples_seen counter, 
    uploads the new file, and returns the new samples_seen count.
    
    If samples_delta is 0, it simply reads the current state and uploads it without modifying samples_seen 
    (though it updates the timestamp).
    """
    
    gcs_blob_base_dir = f"{gcs_prefix}/core_{core_id}"
    local_ckpt_path = os.path.join(local_dir, f"{gcs_prefix}_core_{core_id}_{CHECKPOINT_FILENAME}")
    gcs_blob_path = os.path.join(gcs_blob_base_dir, CHECKPOINT_FILENAME)
    
    # 1. READ: Attempt to download existing checkpoint
    download_success = download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path)

    # 2. LOAD/INITIALIZE: Load or create the progress data
    progress = None
    if download_success:
        try:
            with open(local_ckpt_path, "r") as f:
                progress = json.load(f)
        except Exception:
            print(f"[Core {core_id}] Warning: Checkpoint file corrupted or failed to load. Initializing state.", flush=True)
    
    if progress is None or "samples_seen" not in progress:
        progress = {"core_id": core_id, "samples_seen": 0}
        print(f"[Core {core_id}] Initializing checkpoint state to samples_seen: 0.", flush=True)

    # 3. MODIFY: Update the checkpoint state
    original_samples = progress["samples_seen"]
    
    if samples_delta != 0:
        progress["samples_seen"] += samples_delta
        print(f"[Core {core_id}] Modifying samples: {original_samples} + {samples_delta} = {progress['samples_seen']}", flush=True)
    
    progress["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # 4. WRITE & UPLOAD: Save locally and upload
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


# --- Phase 1: General Access Test (Read-Add-Subtract) ---
def run_phase_1_access_test(core_id, local_dir):
    """Performs a sequence of Read-Modify-Write operations using the general function."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 1: General Access Test (Read/Add/Subtract)\n{'='*50}", flush=True)
    
    # --- Step 1: Read Initial Samples (Delta 0) ---
    print(f"[Core {core_id}] Step 1: Reading initial samples_seen (Expected: 0 if initialized or previous run was successful)")
    try:
        new_samples = access_and_modify_checkpoint(core_id, local_dir, GCS_PREFIX_PHASE_1, samples_delta=0)
        print(f"[Core {core_id}] Result 1: Samples seen is currently {new_samples}")
    except RuntimeError as e:
        print(f"[Core {core_id}] ❌ Step 1 FAILED: {e}", flush=True)
        return # Stop execution if initial read fails

    # --- Step 2: Add 100 Samples (Delta +100) ---
    print(f"\n[Core {core_id}] Step 2: Adding 100 samples and reading back (Expected: 100)")
    try:
        new_samples = access_and_modify_checkpoint(core_id, local_dir, GCS_PREFIX_PHASE_1, samples_delta=100)
        print(f"[Core {core_id}] Result 2: Samples seen after adding 100 is {new_samples}")
        if new_samples == 100:
            print(f"[Core {core_id}] ✅ Step 2 Success: Samples counter correctly updated to 100.")
        else:
            print(f"[Core {core_id}] ⚠️ Step 2 Warning: Samples counter expected 100, got {new_samples}.")
    except RuntimeError as e:
        print(f"[Core {core_id}] ❌ Step 2 FAILED: {e}", flush=True)
        return

    # --- Step 3: Subtract 100 Samples (Delta -100) ---
    print(f"\n[Core {core_id}] Step 3: Subtracting 100 samples and reading back (Expected: 0)")
    try:
        new_samples = access_and_modify_checkpoint(core_id, local_dir, GCS_PREFIX_PHASE_1, samples_delta=-100)
        print(f"[Core {core_id}] Result 3: Samples seen after subtracting 100 is {new_samples}")
        if new_samples == 0:
            print(f"[Core {core_id}] ✅ Step 3 Success: Samples counter correctly reset to 0.")
        else:
            print(f"[Core {core_id}] ❌ Step 3 FAILED: Samples counter expected 0, got {new_samples}.")
    except RuntimeError as e:
        print(f"[Core {core_id}] ❌ Step 3 FAILED: {e}", flush=True)
        return
        
    print(f"\n[Core {core_id}] ✅ Phase 1 Test Sequence Complete.")


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


# --- Phase 3: Verify Access to 32 New Checkpoints (Sharded) ---
def run_access_verification_test(core_id, total_cores, local_dir):
    """Verifies that the 32 newly created files can be accessed and contain 'samples_seen': 0."""
    
    print(f"\n{'='*50}\n[Core {core_id}] PHASE 3: Verifying Access to {TOTAL_TARGET_FOLDERS} New Checkpoints\n{'='*50}", flush=True)
    
    # 1. Determine which folders this core is responsible for
    folders_assigned = []
    for i in range(TOTAL_TARGET_FOLDERS):
        # Shard the work: folder 'i' is handled by core 'i % total_cores'
        if i % total_cores == core_id:
            folders_assigned.append(i)
    
    print(f"[Core {core_id}] Responsible for verifying folders: {folders_assigned}", flush=True)

    test_passed_count = 0
    test_failed_count = 0
    
    for folder_index in folders_assigned:
        # Define paths for the new checkpoint location
        gcs_blob_base_dir = f"{GCS_PREFIX_PHASE_2}/core_{folder_index}"
        local_ckpt_path = os.path.join(local_dir, f"verification_core_{folder_index}_{CHECKPOINT_FILENAME}")
        gcs_blob_path = os.path.join(gcs_blob_base_dir, CHECKPOINT_FILENAME)
        
        # 2. Attempt to download
        download_success = download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path)

        if not download_success:
            print(f"[Core {core_id}] ❌ Verification FAILED for Folder {folder_index}: File not found or download failed.", flush=True)
            test_failed_count += 1
            continue

        # 3. Verify content
        try:
            with open(local_ckpt_path, "r") as f:
                progress = json.load(f)
            
            # Check for core_id consistency and samples_seen value
            if progress.get("core_id") == folder_index and progress.get("samples_seen") == 0:
                print(f"[Core {core_id}] ✅ Verification PASS for Folder {folder_index}. Content confirmed (samples_seen: 0).", flush=True)
                test_passed_count += 1
            else:
                print(f"[Core {core_id}] ❌ Verification FAILED for Folder {folder_index}: Content mismatch. ID: {progress.get('core_id')}, Samples: {progress.get('samples_seen')}", flush=True)
                test_failed_count += 1
                
        except Exception as e:
            print(f"[Core {core_id}] ❌ Verification FAILED for Folder {folder_index}: Error reading content: {e}", flush=True)
            test_failed_count += 1
            
        # Clean up local file
        os.remove(local_ckpt_path)

    # 4. Report results
    print(f"\n[Core {core_id}] PHASE 3 RESULTS: Passed {test_passed_count}/{len(folders_assigned)}. Failed {test_failed_count}.", flush=True)


# --- Worker function for each TPU core ---
def _mp_fn(index):
    core_id = xm.get_ordinal()
    total_cores = xm.xrt_world_size()
    
    # Create a unique local temporary directory for this core
    local_dir = f"/tmp/tpu_ckpt_test_{core_id}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[Core {core_id}] Using device {xm.torch_xla.device()} / Total TPU cores: {total_cores}", flush=True)

    # --- 1. PHASE 1: READ-MODIFY-WRITE TEST ---
    run_phase_1_access_test(core_id, local_dir)
    
    # Synchronize all cores before starting the second phase
    xm.rendezvous("phase_1_complete")

    # --- 2. PHASE 2: CREATE 32 NEW CHECKPOINTS ---
    initialize_32_checkpoints(core_id, total_cores, local_dir)
    
    # Synchronize all cores before starting the verification phase
    xm.rendezvous("phase_2_complete")

    # --- 3. PHASE 3: VERIFY ACCESS TEST ---
    run_access_verification_test(core_id, total_cores, local_dir)
    
    # Final Synchronization
    xm.rendezvous("phase_3_complete")
    
    # Clean up the core's temporary directory
    # os.rmdir(local_dir)
    
    print(f"[Core {core_id}] ALL PHASES COMPLETE.", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    # Ensure all necessary imports are available and authenticated for GCS access
    # This must be run with the appropriate launcher for PyTorch XLA on TPU, e.g.,
    # python -m torch_xla.distributed.xla_multiprocessing --workers=all tpu_checkpoint_sequencing_test.py
    xmp.spawn(_mp_fn, args=())