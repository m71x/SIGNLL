# controller_data_prep_xla.py
import numpy as np
import os
import io
import sys
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import time
from typing import Dict, Optional
import torch

# NOTE: This code requires the 'google-cloud-storage' library to be installed:
# pip install google-cloud-storage

# --- Configuration Constants ---
BUCKET_NAME = "encoder-models-2"
# Base prefix where the core folders (core_0/, core_1/, etc.) reside
GCS_BASE_PREFIX = "siebert-data/siebert-actual-data" 

# --- Global Data Loading Configuration (Flags) ---
FLAGS = {
    # Data loading per core
    "chunk_filename": "embeddings_chunk_0.npz", 
    "samples_to_load": 19500, 
    # Use different filenames for different cores, e.g., in a loop
    # If the files are sharded by core ID, 'embeddings_chunk_0.npz' might be correct for all.
}


# --- GCS Client Setup (IMPORTANT) ---
try:
    from google.cloud import storage
    # Initialize the client. This typically handles authentication automatically
    # NOTE: GCS_CLIENT must be initialized *outside* the xmp.spawn function if possible,
    # or ensure it's thread/process safe if initialized inside. Here we initialize globally.
    GCS_CLIENT = storage.Client()
except ImportError:
    print("ERROR: 'google-cloud-storage' library not found. Please install it.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to initialize Google Cloud Storage client: {e}")
    sys.exit(1)


def load_npz_from_gcs(core_id: int, filename: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Downloads and loads sharded NPZ file from GCS for a specific core.
    """
    # 1. Construct the full GCS Blob Path
    # The path is constructed to read from the folder specific to the core ID.
    blob_name = f"{GCS_BASE_PREFIX}/core_{core_id}/{filename}"
    
    # Use xm.master_print to only log the full path from the master core (core 0)
    # to avoid spamming the log, but print the Core ID's actions from the core itself.
    if core_id == 0:
        xm.master_print("-" * 60)
        xm.master_print(f"Target Bucket: {BUCKET_NAME}")
        xm.master_print(f"Target Blob Prefix: {blob_name.rsplit('/', 1)[0]}/...")
        xm.master_print("-" * 60)

    try:
        # 2. Get the Blob and Download to Memory
        bucket = GCS_CLIENT.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            print(f"[Core {core_id}] ❌ ERROR: Blob not found at gs://{BUCKET_NAME}/{blob_name}")
            return None

        # Create an in-memory binary stream (BytesIO)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0) # Rewind the buffer to the beginning
        
        buffer_size = buffer.getbuffer().nbytes
        print(f"[Core {core_id}] ✅ GCS Download successful. Data loaded ({buffer_size} bytes).")

        if buffer_size == 0:
            print(f"[Core {core_id}] ❌ CRITICAL ERROR: Downloaded file is EMPTY (0 bytes).")
            return None
        
        # 3. Load NPZ data from the memory buffer
        npz_data = np.load(buffer, allow_pickle=False)

        # 4. Return the content
        return dict(npz_data)

    except Exception as e:
        print(f"[Core {core_id}] FATAL ERROR: GCS/NPZ processing failed: {e}")
        return None

def training_data_download(core_id: int, filename: str, max_entries: int) -> Optional[Dict[str, np.ndarray]]:
    """
    Downloads NPZ data from GCS, slices, and shuffles the first N entries.
    """
    # 1. Load the data
    data = load_npz_from_gcs(core_id, filename)
    if data is None:
        return None

    cls_tokens = data['all_layer_cls_tokens']
    classifications = data['classifications']
    total_samples = cls_tokens.shape[0]

    # 2. Determine the slice size
    N = min(total_samples, max_entries)
    
    print(f"[Core {core_id}] Data Prep: Total samples found: {total_samples}, Slicing to N: {N}")

    if N == 0:
        print(f"[Core {core_id}] ❌ Data slice resulted in 0 samples.")
        return None
    
    # 3. Apply shuffle and slice
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    shuffled_and_sliced_indices = indices[:N]

    shuffled_cls_tokens = cls_tokens[shuffled_and_sliced_indices]
    shuffled_classifications = classifications[shuffled_and_sliced_indices]
    
    print(f"[Core {core_id}] ✅ Data successfully shuffled and sliced. Final samples: {shuffled_cls_tokens.shape[0]}")
    
    # 4. Return the shuffled data
    return {
        'all_layer_cls_tokens': shuffled_cls_tokens,
        'classifications': shuffled_classifications
    }

# ----------------------------------------------------------------------
# Core Function for XLA Multiprocessing
# ----------------------------------------------------------------------

def data_prep_fn(rank, flags):
    """
    The function executed by each TPU core.
    The 'rank' argument is the unique core ID (0 to N-1).
    """
    # 1. Get Core ID and synchronize start
    core_id = xm.get_ordinal()
    num_cores = xm.xrt_world_size()
    
    if core_id == 0:
        xm.master_print(f"🚀 Starting data loading across {num_cores} TPU cores.")
        
    # Synchronization point: wait for all cores to start before loading data
    xm.rendezvous('data_loading_start')
    start_time = time.time()

    # 2. Load and Prepare Data
    # Each core loads its own shard based on its unique core_id
    training_data = training_data_download(
        core_id=core_id, 
        filename=flags["chunk_filename"], 
        max_entries=flags["samples_to_load"]
    )
    
    # 3. Log Results
    if training_data:
        cls_tokens_array = training_data['all_layer_cls_tokens']
        classifications_array = training_data['classifications']
        num_samples = cls_tokens_array.shape[0]
        
        # Move the data to the XLA device (important for subsequent training)
        # We only do this check for a sanity test
        device = xm.xla_device()
        cls_tokens_tensor = torch.from_numpy(cls_tokens_array).float().to(device)
        classifications_tensor = torch.from_numpy(classifications_array).long().to(device)
        
        print(f"[Core {core_id}] 🎉 SUCCESS: Loaded {num_samples} samples. Data moved to {device}.")
        # Use master_print for high-level summaries to keep logs clean
        if core_id == 0:
             xm.master_print(f"Sample CLS token shape: {cls_tokens_tensor.shape}")
             xm.master_print(f"Sample Classification: {classifications_tensor[0:5]}")
             
    else:
        print(f"[Core {core_id}] ⚠️ WARNING: Failed to load data or returned 0 samples.")

    # 4. Synchronization and Reporting
    # Wait for ALL cores to finish loading before exiting (prevents process termination race conditions)
    xm.rendezvous('data_loading_end')

    if core_id == 0:
        end_time = time.time()
        xm.master_print(f"✅ Data loading and preparation complete for all cores in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    # Ensure all cores use the same random seed for consistent shuffling
    # Note: For production use, you might use a more sophisticated mechanism.
    np.random.seed(42) 
    
    # Execute the data_prep_fn on all available TPU cores
    xmp.spawn(data_prep_fn, args=(FLAGS,), start_method='fork')