import os
import time
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import gcld3
# Import the REAL GCS function
from gcs_access import upload_file_to_gcs

# --- Configuration Constants (Updated for ALL-LAYER extraction) ---
LOCAL_MODEL_PATH = "/home/mikexi/siebert_model"
# CRITICAL: Changing the output prefix because the data structure is now 3D.
UPLOAD_PREFIX = "siebert-data/siebert-actual-data" 
BUCKET_NAME = "encoder-models-2"

# Efficiency Parameters
INFERENCE_BATCH_SIZE = 64
IO_ACCUMULATION_THRESHOLD = 20000 


# --- Utility Functions (Simulated External Dependencies) ---

def get_core_ordinal():
    """Returns the XLA core ID for distributed logging."""
    try:
        return xm.get_ordinal()
    except Exception:
        return 0

# --- gcld3 Check Function (Based on File 2's logic) ---

def check_gcld3_functionality():
    """
    Performs a simple language detection test using gcld3 to ensure the 
    library is working correctly within the current environment.
    """
    core_id = get_core_ordinal()
    print(f"[Core {core_id}] Starting gcld3 functionality check...", flush=True)
    try:
        detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        test_text = "The quick brown fox jumps over the lazy dog."
        result = detector.FindLanguage(text=test_text)
        
        if result.is_reliable and result.language == 'en':
            print(f"[Core {core_id}] ✅ gcld3 check passed. Detected '{result.language}' reliably.", flush=True)
            return True
        else:
            print(f"[Core {core_id}] ❌ gcld3 check failed. Result: {result.language}, Reliable: {result.is_reliable}", flush=True)
            return False
    except Exception as e:
        print(f"[Core {core_id}] ❌ gcld3 check failed with exception: {e}", flush=True)
        return False


def is_english_and_reliable(text, detector):
    """Checks if the text is reliably identified as English."""
    if not text or not isinstance(text, str):
        return False
    result = detector.FindLanguage(text=text)
    return result.is_reliable and result.language == 'en'


# --- Core Inference Function (MODIFIED) ---

def run_inference_and_store_cls_tokens(model, tokenizer, core_id, text_samples, max_seq_length):
    """
    Tokenizes a batch of text, runs the model, and extracts the 
    CLS token from *all* layers along with the final classification index (0 or 1).

    Args:
        model: The loaded Siebert model.
        tokenizer: The loaded tokenizer.
        core_id (int): The ID of the current core.
        text_samples (list[str]): A list of text samples to process.
        max_seq_length (int): The maximum sequence length for tokenization.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            1. All-layer CLS tokens: Shape (Batch, Num_Layers, Hidden_Dim)
            2. Class indices: Shape (Batch,)
    """
    
    # NOTE: It is critical that 'model' was loaded with output_hidden_states=True 
    # in main.py for this function to work correctly.

    device = xm.torch_xla.device()

    # Tokenization with fixed max_length padding
    inputs = tokenizer(
        text_samples, 
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        
    # --- 1. Extract CLS token from EVERY hidden state ---
    # outputs.hidden_states is a tuple/list where the first element is the 
    # embedding output, and subsequent elements are layer outputs.
    all_layer_cls_tokens = [
        # tensor[:, 0, :] extracts the CLS token (index 0) from the sequence dimension
        t[:, 0, :].to(torch.float32).cpu().numpy()
        for t in outputs.hidden_states
    ]
    
    # Stack the CLS tokens from all layers together into one 3D array
    # Resulting Shape: (Batch_Size, Num_Layers, Hidden_Dim)
    stacked_cls_tokens = np.stack(all_layer_cls_tokens, axis=1)

    # --- 2. Extract the final prediction (0 or 1) ---
    # Logits are (Batch, 2). Argmax gives the predicted class index (0 or 1).
    class_indices_np = torch.argmax(outputs.logits, dim=1).cpu().numpy().astype(np.uint8)

    return stacked_cls_tokens, class_indices_np


# --- I/O Write-Out Function (MODIFIED) ---

def run_io_write_out(core_id, accumulated_all_layer_cls_tokens, accumulated_classifications, chunk_index, local_dir):
    """
    Concatenates accumulated outputs (All-Layer CLS Tokens and Classifications), 
    saves them to a compressed NumPy file locally, and uploads the file to GCS.

    Args:
        core_id (int): The ID of the current core.
        accumulated_all_layer_cls_tokens (list[np.ndarray]): List of 3D CLS token batches.
        accumulated_classifications (list[np.ndarray]): List of 1D classification batches (0/1).
        chunk_index (int): The current sequential index for the chunk file name.
        local_dir (str): Temporary local directory for saving.
    """
    
    print(f"[Core {core_id}] 🔥 I/O THRESHOLD REACHED. Writing chunk {chunk_index}...", flush=True)

    # 1. Concatenate all accumulated arrays
    # This concatenates the 3D arrays along the batch dimension (axis=0)
    all_cls_tokens = np.concatenate(accumulated_all_layer_cls_tokens, axis=0)
    # The classification indices are 1D arrays
    all_classifications = np.concatenate(accumulated_classifications, axis=0)

    # 2. Save locally using NumPy
    local_path = os.path.join(local_dir, f"embeddings_chunk_{chunk_index}.npz")
    
    # Save the 3D CLS token vectors and the final classifications together.
    np.savez_compressed(local_path, 
                        # New key for clarity: reflects that this is the full 3D data
                        all_layer_cls_tokens=all_cls_tokens, 
                        classifications=all_classifications.astype(np.uint8))
    
    # The shape will now be (N_samples, N_layers, N_dim) - e.g. (20000, 13, 768)
    print(f"[Core {core_id}] Saved {all_cls_tokens.shape[0]} samples locally to {local_path}. Data shape: {all_cls_tokens.shape}", flush=True)

    # 3. Upload file to GCS
    gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/embeddings_chunk_{chunk_index}.npz"
    upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
    
    # 4. Cleanup
    if os.path.exists(local_path):
        os.remove(local_path)