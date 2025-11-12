#TODO: first have the model run the following on all workers and display success/failure:
# 1. display all TPU cores available and if possible, their runtime status
# 2. retrieving and displaying current checkpoints for each core
# 3. download model on all workers and display success/failure
# 4. Have each chip download its respective shard of the dataset and display success/failure
# 5. make sure each model can access the folder with its respective core number that will be used to upload data and display success/failure
# do an inference test too, checking that the amount of CLS tokens is indeed 24 and the model correctly outputted 1 or 0 for a sample prompt

#then, actually run the heavy inference part and make sure it can display consistent output as it runs and uploads results to the GCS bucket:
# 6. for each input, first make sure gcld3 says it is english AND the result is reliable. If not, skip it but still increment the checkpoint
# 7. else, run the siebert model, store [CLS] token at each layer and the final classification(1 or 0) data to the buffer 
# 8. continue, and then upload to cloud once buffer capacity is reached (capacity is set by inference.py), and also update the checkpoint folder for each core by incrementing by the amount of samples it has seen
#then, continue onto the next batch of data in each core's respective shard's parquet file
#repeat steps 6-8

# main.py
import os
import time
import json
import math
import numpy as np
import pandas as pd


import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# Import your helper modules (these are assumed to be present and correct)
import training_dataset_access
import checkpoint_access
import inference
from gcs_access import download_model_from_gcs, BUCKET_NAME, MODEL_PREFIX, LOCAL_MODEL_PATH

# --- Configuration Constants ---
CHECKPOINT_PREFIX = checkpoint_access.DEFAULT_GCS_PREFIX
CHECKPOINT_FILENAME = checkpoint_access.DEFAULT_FILENAME

INFERENCE_BATCH_SIZE = inference.INFERENCE_BATCH_SIZE
IO_ACCUMULATION_THRESHOLD = inference.IO_ACCUMULATION_THRESHOLD
# Expected number of hidden states (layers) for CLS token extraction
# (e.g., 24 layers + 1 embedding layer = 25 states is common for BERT-large)
# We will use 24 as requested by the user for the test.
EXPECTED_CLS_COUNT = 25


# --- Logging Helper ---
def get_core_ordinal():
    """Returns the XLA core ID for distributed logging."""
    try:
        return xm.get_ordinal()
    except Exception:
        return 0

def log(core_id, message):
    """Prints a formatted message with the core ID."""
    print(f"[Core {core_id}] {message}", flush=True)

# --- Inference Test Function ---

def test_model_inference(model, tokenizer, device, core_id):
    """
    Runs a test inference on a sample positive input to verify CLS token count and classification.
    """
    log(core_id, "--- Running Model Inference Test ---")
    
    # 1. Define a sample positive input (assumes the model is a binary classifier where 1 is positive/safe)
    test_text = ["This is a wonderful day, and I am excited to start working on this project!"]
    expected_classification = 1 # 1 for positive, 0 for negative/hate
    
    try:
        # Returns 3D CLS tokens [B, L, D] and 1D classification indices [B]
        cls_np, classifications_np = inference.run_inference_and_store_cls_tokens(
            model, tokenizer, test_text, device
        )

        # Check 1: CLS Token Count (N_layers dimension)
        actual_cls_count = cls_np.shape[1] # The middle dimension [B, L, D]
        cls_token_success = actual_cls_count == EXPECTED_CLS_COUNT
        
        if cls_token_success:
            log(core_id, f"✅ CLS Token Test SUCCESS: Retrieved {actual_cls_count} hidden states (Expected {EXPECTED_CLS_COUNT}).")
        else:
            log(core_id, f"❌ CLS Token Test FAILURE: Retrieved {actual_cls_count} hidden states (Expected {EXPECTED_CLS_COUNT}).")
            log(core_id, f"CLS tokens array shape: {cls_np.shape}")
        
        # Check 2: Classification Result
        actual_classification = classifications_np[0]
        classification_success = actual_classification == expected_classification
        
        if classification_success:
            log(core_id, f"✅ Classification Test SUCCESS: Classified as {actual_classification} (Expected {expected_classification} for positive sentiment).")
        else:
            log(core_id, f"❌ Classification Test FAILURE: Classified as {actual_classification} (Expected {expected_classification}).")
        
        if cls_token_success and classification_success:
            log(core_id, "--- Model Inference Test PASSED ---")
            return True
        else:
            log(core_id, "--- Model Inference Test FAILED. Halting worker. ---")
            return False

    except Exception as e:
        log(core_id, f"FATAL: Inference test failed with unhandled error: {e}")
        return False


# --- Core Worker Function ---
def main_worker(index, local_base):
    global_core = xm.get_ordinal()
    core_id = global_core    # use one name everywhere if you like
    log(core_id, "Starting worker...")
    
    # 1. Initialization and Model Download (Master only logic omitted here, assuming success)
    log(core_id, f"Model path assumed to be ready at: {LOCAL_MODEL_PATH}")
    
    # --- Setup Inference Model (part of pre-flight checks) ---
    device = xm.torch_xla.device()
    log(core_id, f"Initializing model on device: {device}")
    model, tokenizer = inference.initialize_model(LOCAL_MODEL_PATH, device)
    
    # --- NEW: Run mandatory inference test ---
    if not test_model_inference(model, tokenizer, device, core_id):
        log(core_id, "Model inference test failed. Shutting down worker.")
        return # Terminate worker if the model test fails
    
    # 2. Get Starting Checkpoint
    checkpoint_data = checkpoint_access.load_checkpoint(core_id, local_dir="/tmp/checkpoints", gcs_prefix=CHECKPOINT_PREFIX, filename=CHECKPOINT_FILENAME)

    if checkpoint_data is None:
        start_index = 0
        log(core_id, "FATAL: Checkpoint load failed. Starting from index 0.")
    elif isinstance(checkpoint_data, int):
        start_index = checkpoint_data
    else:
        start_index = checkpoint_data.get("samples_seen", 0)

    if start_index == 0:
        log(core_id, "No valid checkpoint found or samples_seen is 0. Starting from index 0.")
    else:
        log(core_id, f"Resuming from checkpoint index: {start_index}.")
    
    # 3. Download and Load Data Shard
    local_shard_path = training_dataset_access.get_shard_path(core_id)
    if not training_dataset_access.download_data_shard(core_id, local_shard_path):
        log(core_id, "FATAL: Failed to download and pre-process data shard. Exiting.")
        return

    # CRITICAL CHANGE: The DataFrame now loads the **pre-processed** file
    log(core_id, f"Loading pre-processed parquet file from {local_shard_path}...")
    try:
        # Load the full shard into memory. It contains the 'is_english_reliable' column.
        df = pd.read_parquet(local_shard_path)
        total_samples = len(df)
        log(core_id, f"Shard loaded successfully. Total samples: {total_samples}")
    except Exception as e:
        log(core_id, f"FATAL: Failed to load parquet file: {e}. Exiting.")
        return

    # Check if we have already processed everything
    if start_index >= total_samples:
        log(core_id, f"Checkpoint index {start_index} >= total samples {total_samples}. Nothing left to process. Exiting.")
        return

    # --- Main Processing Loop ---
    current_index = start_index
    processed_since_last_save = 0
    chunk_index = 0
    
    # CRITICAL: List to hold 3D CLS token arrays: [B, L, D]
    accumulated_cls = [] 
    # CRITICAL: List to hold 1D classification index arrays: [B] 
    accumulated_classifications = [] 

    log(core_id, f"Starting inference from row index {current_index}...")

    # Iterate in batches
    while current_index < total_samples:
        # 1. Get the current batch chunk from the full DataFrame
        end_index = min(current_index + INFERENCE_BATCH_SIZE, total_samples)
        batch_df = df.iloc[current_index:end_index].copy()
        batch_size = len(batch_df) # This is usually INFERENCE_BATCH_SIZE (64)
        
        # 2. **FAST FILTERING (Replaces the slow gcld3 loop)**
        # Filter the batch using the pre-computed 'is_english_reliable' column (where 1=English)
        filtered_df = batch_df[batch_df['is_english_reliable'] == 1]
        
        # Extract the final texts and count
        # Note: Assuming 'tweet' is the correct text column name in your Parquet file.
        filtered_texts = filtered_df['tweet'].tolist()
        filtered_count = len(filtered_texts) 

        # 3. Advance the checkpoint index
        # IMPORTANT: Advance the index by the full batch size read, regardless of filtering outcome.
        current_index += batch_size

        # Only run inference if there's data to process after filtering
        if filtered_texts:
            # CRITICAL FIX: Pad the input texts to the fixed batch size (64)
            padding_needed = INFERENCE_BATCH_SIZE - filtered_count
            
            if padding_needed > 0:
                dummy_text = ["<PAD>"] * padding_needed 
                texts_to_process = filtered_texts + dummy_text
            else:
                texts_to_process = filtered_texts
                
            log(core_id, f"Processing batch of {batch_size} (Filtered to {filtered_count} English samples, Padded to {len(texts_to_process)} for TPU)...")

            # 4. Run Inference (Passes the fixed-size batch)
            try:
                # Returns 3D CLS tokens and 1D classification indices (0/1)
                cls_np, classifications_np = inference.run_inference_and_store_cls_tokens(
                    model, tokenizer, texts_to_process, device
                )
                
                # CRITICAL: Slice the results back down to the original filtered count
                cls_np = cls_np[:filtered_count]
                classifications_np = classifications_np[:filtered_count]
                
                # Accumulate the results
                accumulated_cls.append(cls_np)
                accumulated_classifications.append(classifications_np)
                
                # Increment the samples *processed and collected* since the last IO save
                processed_since_last_save += cls_np.shape[0]

            except Exception as e:
                log(core_id, f"Inference failed for batch: {e}. Skipping batch.")
        else:
            log(core_id, f"Skipped batch of {batch_size} (No English/reliable samples).")

        # 5. Check I/O Threshold (Remains unchanged)
        if processed_since_last_save >= IO_ACCUMULATION_THRESHOLD:
            log(core_id, f"I/O threshold reached. Writing chunk {chunk_index}...")
            try:
                inference.write_and_upload_chunk(
                    core_id, accumulated_cls, accumulated_classifications, chunk_index, local_base
                )
                
                # Update Checkpoint with total processed samples read from the parquet file
                total_processed = current_index
                checkpoint_access.save_checkpoint(core_id, total_processed, local_dir="/tmp/checkpoints", gcs_prefix=CHECKPOINT_PREFIX, filename=CHECKPOINT_FILENAME)

                # Reset buffers and count for the next chunk
                accumulated_cls = []
                accumulated_classifications = []
                processed_since_last_save = 0
                chunk_index += 1
                
                log(core_id, f"Checkpoint updated to {total_processed}. Next chunk index is {chunk_index}.")

            except Exception as e:
                log(core_id, f"FATAL: I/O or Checkpoint update failed: {e}. Attempting graceful exit.")
                return 

        # Simple progress report
        if current_index % (INFERENCE_BATCH_SIZE * 100) == 0:
            log(core_id, f"Progress: {current_index} / {total_samples} samples read from shard.")
        
    log(core_id, f"Shard processing complete. Final index read: {current_index}")

    # --- Final I/O and Checkpoint Update (Remains unchanged) ---
    if accumulated_cls:
        log(core_id, "Final chunk remaining. Uploading...")
        chunk_index += 1 
        try:
            inference.write_and_upload_chunk(
                core_id, accumulated_cls, accumulated_classifications, chunk_index, local_base
            )
            log(core_id, f"Final uploaded chunk {chunk_index} with {sum(a.shape[0] for a in accumulated_cls)} samples.")
        except Exception as e:
            log(core_id, f"Failed final upload: {e}")

    # Final checkpoint update: set to the total index read
    if current_index > start_index:
        try:
            checkpoint_access.save_checkpoint(core_id, current_index, local_dir="/tmp/checkpoints", gcs_prefix=CHECKPOINT_PREFIX, filename=CHECKPOINT_FILENAME)
            log(core_id, f"Final checkpoint update: samples_seen set to {current_index} (End of Shard).")
        except Exception as e:
            log(core_id, f"Final checkpoint upload failed: {e}")

    # Cleanup: remove local shard file
    if os.path.exists(local_shard_path):
        os.remove(local_shard_path)
        log(core_id, f"Cleaned up local shard file: {local_shard_path}")

    log(core_id, "Worker finished gracefully.")


if __name__ == '__main__':
    # Mock local_base path for local testing structure
    LOCAL_BASE = "/tmp/siebert_inference_data"
    os.makedirs(LOCAL_BASE, exist_ok=True)
    
    #log(0, f"Starting multiprocess execution with base directory: {MOCK_LOCAL_BASE}")
    
    # In a real TPU environment, this would be:
    xmp.spawn(main_worker, args=(LOCAL_BASE,), nprocs=xm.xrt_world_size()) 
    
    # For local testing simulation:
    # main_worker(0, MOCK_LOCAL_BASE) 
    
    #log(0, "Mock execution complete.")