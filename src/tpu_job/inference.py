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


# --- Utility Functions ---
def get_core_ordinal():
    """Returns the XLA core ID for distributed logging."""
    try:
        return xm.get_ordinal()
    except Exception:
        return 0

# --- gcld3 Check Function ---
def check_gcld3_functionality():
    """
    Performs a simple language detection test using gcld3 to ensure the 
    library is working correctly within the current environment.
    """
    core_id = get_core_ordinal()
    print(f"[Core {core_id}] Starting gcld3 functionality check...", flush=True)
    try:
        detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        test_text = "This is a test of the gcld3 library."
        result = detector.FindLanguage(text=test_text)
        if result.language == 'en' and result.is_reliable:
            print(f"[Core {core_id}] ✅ gcld3 check SUCCESS: '{test_text}' detected as '{result.language}' (Reliable: {result.is_reliable}).", flush=True)
            return True
        else:
            print(f"[Core {core_id}] ❌ gcld3 check FAILED: Incorrectly detected as '{result.language}'.", flush=True)
            return False
    except Exception as e:
        print(f"[Core {core_id}] ❌ gcld3 check FAILED with error: {e}", flush=True)
        return False


# --- Inference Setup ---

def initialize_model(model_path, device):
    """
    Loads the tokenizer and the sequence classification model onto the specified device (TPU).
    """
    core_id = get_core_ordinal()
    print(f"[Core {core_id}] Initializing model from {model_path}...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model weights and move to the XLA device
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    
    # Put the model into evaluation mode
    model.eval()
    
    print(f"[Core {core_id}] ✅ Model initialization successful.", flush=True)
    return model, tokenizer

# --- Inference Execution ---

def run_inference_and_store_cls_tokens(model, tokenizer, texts, device):
    """
    Tokenizes a list of texts, runs inference on the Siebert model, extracts 
    the all-layer CLS tokens, and calculates the final classification index.

    Args:
        model (PreTrainedModel): The Siebert model.
        tokenizer (PreTrainedTokenizer): The model's tokenizer.
        texts (list[str]): The batch of text strings (max 64).
        device (torch.device): The XLA device to run on.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            1. All-layer CLS tokens (3D: [B, L+1, D])
            2. Classification indices (1D: [B])
    """
    
    core_id = get_core_ordinal()
    
    # 1. Tokenization and Input Creation
    FIXED_MAX_LENGTH = 128
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=FIXED_MAX_LENGTH
    )

    
    # Move inputs to the XLA device
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    
    # 2. Inference and Output Extraction
    with torch.no_grad():
        # CRITICAL: Request hidden states to get all layer outputs
        outputs = model(**inputs, output_hidden_states=True)
        # Get logits (shape [B, 2])
        logits = outputs.logits 
        
        # Get hidden states (tuple of Tensors, one for each layer + embedding)
        hidden_states = outputs.hidden_states 
        
        # 3. Optimized Processing: All-layer CLS Tokens
        # Stack the CLS token (index 0) from all hidden states (25 layers)
        # using PyTorch on the TPU, then transfer the final tensor once.
        all_cls_tokens_tensor = torch.stack(
            [state[:, 0, :] for state in hidden_states], # Select CLS token (index 0) from each layer
            dim=1 # Stack along the layer dimension to get [B, L+1, D]
        )
        
        # Convert the final stacked tensor to NumPy on the CPU
        cls_np = all_cls_tokens_tensor.cpu().numpy()

        # 4. Optimized Processing: Classification Indices
        # Use torch.argmax on the TPU, then transfer the result once.
        classifications_tensor = torch.argmax(logits, dim=1)
        classifications_np = classifications_tensor.cpu().numpy()

    # Use xm.mark_step() to synchronize execution on the TPU core
    xm.torch_xla.sync()
    
    print(f"[Core {core_id}] Inference complete. All-layer CLS shape: {cls_np.shape}", flush=True)

    return cls_np, classifications_np

# --- I/O Operations ---

def write_and_upload_chunk(core_id, accumulated_all_layer_cls_tokens, accumulated_classifications, chunk_index, local_dir):
    """
    Concatenates accumulated outputs (CLS Tokens and Classifications), 
    saves them to a compressed NumPy file locally, and uploads the file to GCS.
    
    Args:
        core_id (int): The ID of the current core.
        accumulated_all_layer_cls_tokens (list[np.ndarray]): List of 3D CLS token batches.
        accumulated_classifications (list[np.ndarray]): List of 1D classification index batches (0/1).
        chunk_index (int): The current sequential index for the chunk file name.
        local_dir (str): Temporary local directory for saving.
    """
    
    print(f"[Core {core_id}] 🔥 I/O THRESHOLD REACHED. Writing chunk {chunk_index}... (Target: {IO_ACCUMULATION_THRESHOLD})", flush=True)

    # 1. Concatenate all accumulated arrays
    # This concatenates the 3D arrays along the batch dimension (axis=0)
    all_cls_tokens = np.concatenate(accumulated_all_layer_cls_tokens, axis=0)
    # The classification indices are 1D arrays
    all_classifications = np.concatenate(accumulated_classifications, axis=0)

    # 2. Save locally using NumPy
    local_path = os.path.join(local_dir, f"embeddings_chunk_{chunk_index}.npz")
    
    # Save the 3D CLS token vectors and the final classifications together.
    np.savez_compressed(local_path, 
                        # Key 1: 3D CLS data (N_samples, N_layers, N_dim)
                        all_layer_cls_tokens=all_cls_tokens, 
                        # Key 2: 1D Classification Index (0 or 1)
                        classifications=all_classifications.astype(np.uint8))
    
    print(f"[Core {core_id}] Saved {all_cls_tokens.shape[0]} samples locally to {local_path}. Data shape: {all_cls_tokens.shape}", flush=True)

    # 3. Upload file to GCS
    gcs_blob_path = f"{UPLOAD_PREFIX}/core_{core_id}/embeddings_chunk_{chunk_index}.npz"
    upload_file_to_gcs(BUCKET_NAME, local_path, gcs_blob_path)
    
    # 4. Cleanup
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"[Core {core_id}] Cleaned up local file: {local_path}", flush=True)