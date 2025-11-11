import os
import time
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import gcld3

# --- Configuration Constants (Mirroring File 1) ---
LOCAL_MODEL_PATH = "/home/mikexi/siebert_model"
UPLOAD_PREFIX = "siebert-data/siebert-data-cls-only"
BUCKET_NAME = "encoder-models-2"

# Efficiency Parameters
INFERENCE_BATCH_SIZE = 64
IO_ACCUMULATION_THRESHOLD = 5000 


# --- Utility Functions (Simulated External Dependencies) ---

def get_core_ordinal():
    """Returns the XLA core ID for distributed logging."""
    try:
        return xm.get_ordinal()
    except Exception:
        return 0

def upload_file_to_gcs(bucket_name, local_path, gcs_blob_path):
    """
    [Placeholder function for GCS upload]
    In a real pipeline, this would be imported from a utility module.
    """
    core_id = get_core_ordinal()
    print(f"[Core {core_id}] [MOCK GCS UPLOAD] Simulating upload of {local_path} to gs://{bucket_name}/{gcs_blob_path}...", flush=True)
    # Simulate success
    return True


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
        test_text = "This is a test sentence in English, the language should be 'en'."
        result = detector.FindLanguage(text=test_text)
        
        if result.language == 'en' and result.is_reliable:
            print(f"[Core {core_id}] ✅ gcld3 check passed: Detected '{result.language}' with confidence {result.probability:.2f}.", flush=True)
            return True
        else:
            print(f"[Core {core_id}] ❌ gcld3 check failed: Expected 'en', got '{result.language}'.", flush=True)
            return False
    except Exception as e:
        print(f"[Core {core_id}] ❌ gcld3 check failed with error: {e}", flush=True)
        return False


# --- Model Setup Function ---

def setup_model_and_tokenizer(local_model_path, device):
    """
    Initializes the tokenizer and model, loading it to the XLA device.
    Assumes the model is already downloaded locally.

    Returns:
        tuple: (tokenizer, model, max_seq_length)
    """
    core_id = get_core_ordinal()
    
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path,
        # Use dtype for performance, ensure output_hidden_states is True
        dtype=torch.bfloat16, 
        output_hidden_states=True
    ).to(device)
    model.eval()
    
    # Determine the effective max sequence length
    MAX_SEQ_LENGTH = tokenizer.model_max_length if tokenizer.model_max_length > 0 and tokenizer.model_max_length <= 512 else 512
    
    print(f"[Core {core_id}] Model setup complete. Max seq length: {MAX_SEQ_LENGTH}", flush=True)
    
    return tokenizer, model, MAX_SEQ_LENGTH


# --- Core Inference and Output Function ---

def run_inference_and_store_cls_tokens(model, tokenizer, device, sample_texts, max_seq_length):
    """
    Runs inference on a batch of text, extracts the CLS token and logits, 
    and returns them as NumPy arrays.

    Args:
        model (PreTrainedModel): The loaded Siebert model.
        tokenizer (PreTrainedTokenizer): The loaded tokenizer.
        device (torch.device): The XLA device to run on.
        sample_texts (list[str]): Batch of text samples.
        max_seq_length (int): The maximum sequence length for padding/truncation.

    Returns:
        tuple: (cls_tokens_np, logits_np)
    """
    core_id = get_core_ordinal()
    
    # 1. Tokenization and Transfer
    inputs = tokenizer(
        sample_texts, 
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    ).to(device)

    # 2. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Data Transfer and Extraction (CLS token and Logits)
    
    # Extract the last layer hidden states (Batch, Sequence Length, Hidden Dim)
    last_hidden_state = outputs.hidden_states[-1] 
    
    # Extract the CLS token, which is the first token in the sequence dimension (index 0)
    # Shape: (Batch Size, Hidden Dim)
    cls_tokens = last_hidden_state[:, 0, :]
    
    # Transfer to CPU and convert to float32 NumPy arrays
    cls_tokens_np = cls_tokens.to(torch.float32).cpu().numpy()
    logits_np = outputs.logits.to(torch.float32).cpu().numpy()
    
    print(f"[Core {core_id}] Ran inference on batch size {len(sample_texts)}. CLS tokens extracted.", flush=True)
    
    # Returns 2D arrays (Batch, Hidden_Dim) and (Batch, Logit_Dim)
    return cls_tokens_np, logits_np


# --- I/O Handling Function ---

def save_and_upload_outputs(core_id, accumulated_cls_tokens, accumulated_logits, chunk_index, local_dir):
    """
    Concatenates accumulated outputs, saves them to a compressed NumPy file 
    locally, and uploads the file to GCS.

    Args:
        core_id (int): The ID of the current core.
        accumulated_cls_tokens (list[np.ndarray]): List of CLS token batches.
        accumulated_logits (list[np.ndarray]): List of logit batches.
        chunk_index (int): The current sequential index for the chunk file name.
        local_dir (str): Temporary local directory for saving.
    """
    
    print(f"[Core {core_id}] 🔥 I/O THRESHOLD REACHED. Writing chunk {chunk_index}...", flush=True)

    # 1. Concatenate all accumulated arrays
    all_cls_tokens = np.concatenate(accumulated_cls_tokens, axis=0)
    all_logits = np.concatenate(accumulated_logits, axis=0)

    # 2. Save locally using NumPy
    local_path = os.path.join(local_dir, f"cls_tokens_chunk_{chunk_index}.npz")
    
    # Save the CLS token vectors and logits together
    np.savez_compressed(local_path, cls_tokens=all_cls_tokens, logits=all_logits)
    
    print(f"[Core {core_id}] Saved {all_cls_tokens.shape[0]} samples locally to {local_path}", flush=True)

    # 3. Upload file to GCS
    gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/cls_tokens_chunk_{chunk_index}.npz"
    upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
    
    # 4. Clean up the local file immediately after upload
    os.remove(local_path)
    
    print(f"[Core {core_id}] Chunk {chunk_index} upload complete.", flush=True)