import sys
import os
import numpy as np

# Import the core loading function from your existing GCS loader script
# Note: This relies on GCS_CLIENT being initialized in gcs_npz_loader.py
try:
    from gcs_npz_loader import load_npz_from_gcs
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import load_npz_from_gcs. Ensure gcs_npz_loader.py is accessible and imports its dependencies correctly. Error: {e}")
    sys.exit(1)


# --- Configuration for Validation ---
# You need to adjust these values to check the file you just uploaded from inference.py
VALIDATION_CORE_ID = 10     # The TPU Core ID that saved the file (e.g., core_10)
VALIDATION_CHUNK_INDEX = 1  # The chunk index of the file you want to validate (e.g., embeddings_chunk_1.npz)
# N_SAMPLES_EXPECTED is the minimum number of samples you expect this chunk to have.
N_SAMPLES_EXPECTED = 1000   


def validate_uploaded_chunk(core_id: int, chunk_index: int, expected_samples: int) -> bool:
    """
    Downloads a specific NPZ chunk from GCS, attempts to load it, and validates its structure.
    
    Args:
        core_id: The core ID (folder name).
        chunk_index: The chunk index (file name suffix).
        expected_samples: Minimum number of samples expected in the chunk.

    Returns:
        True if the file is successfully loaded and passes basic structural checks, False otherwise.
    """
    chunk_filename = f"embeddings_chunk_{chunk_index}.npz"
    print(f"\n--- Starting Validation for {chunk_filename} (Core {core_id}) ---")

    # The loading function handles download, buffer loading, and corruption checks
    data = load_npz_from_gcs(core_id, chunk_filename)
    
    if data is None:
        print(f"\n❌ VALIDATION FAILED: Could not load data from GCS. See error logs above.")
        return False

    # --- Structural Validation ---
    
    try:
        cls_tokens = data['all_layer_cls_tokens']
        classifications = data['classifications']
        
        N_samples = cls_tokens.shape[0]

        # 1. Size Check
        if N_samples < expected_samples:
            print(f"❌ VALIDATION FAILED: Samples loaded ({N_samples}) is less than expected minimum ({expected_samples}).")
            return False
        
        # 2. Shape Check (N, L, D) -> (N, 25, 768)
        if cls_tokens.ndim != 3 or cls_tokens.shape[1] != 25 or cls_tokens.shape[2] != 768:
            print(f"❌ VALIDATION FAILED: CLS token shape is incorrect: {cls_tokens.shape}. Expected (N, 25, 768).")
            return False

        # 3. Dtype Check
        if cls_tokens.dtype != np.float32:
            print(f"⚠️ WARNING: CLS tokens dtype is {cls_tokens.dtype}. Expected float32.")
            
        if classifications.dtype != np.uint8:
            print(f"❌ VALIDATION FAILED: Classification dtype is incorrect: {classifications.dtype}. Expected uint8.")
            return False
            
        # 4. Range Check (Classifications)
        if classifications.min() < 0 or classifications.max() > 1:
            print(f"❌ VALIDATION FAILED: Classification values out of expected [0, 1] range.")
            return False

        print(f"\n✅ VALIDATION SUCCESSFUL: File is valid.")
        print(f"   Loaded Samples: {N_samples}")
        print(f"   CLS Tokens Shape: {cls_tokens.shape}")
        print(f"   Classifications Dtype: {classifications.dtype}")
        return True

    except KeyError as e:
        print(f"\n❌ VALIDATION FAILED: Missing expected key in NPZ file: {e}")
        return False
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: Unexpected error during validation checks: {e}")
        return False


if __name__ == '__main__':
    # You would typically run this script after 'inference.py' completes a chunk upload.
    # Adjust the constants above (VALIDATION_CORE_ID, VALIDATION_CHUNK_INDEX, N_SAMPLES_EXPECTED) 
    # to test the target file.
    
    if validate_uploaded_chunk(VALIDATION_CORE_ID, VALIDATION_CHUNK_INDEX, N_SAMPLES_EXPECTED):
        print("\nFinal Result: The uploaded NPZ file appears structurally sound.")
    else:
        print("\nFinal Result: The uploaded NPZ file is either corrupted or does not meet structural requirements.")