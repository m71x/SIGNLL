#handles downloading training data
#have a function that downloads a sample chunk npz data and verifies things like the dimension of the CLS token and the CLS token and the classification can be seen

import numpy as np
import os
import io
import sys

# NOTE: This code requires the 'google-cloud-storage' library to be installed:
# pip install google-cloud-storage

# --- Configuration Constants (Derived from inference.py) ---
BUCKET_NAME = "encoder-models-2"
# Base prefix where the core folders (core_0/, core_1/, etc.) reside
GCS_BASE_PREFIX = "siebert-data/siebert-actual-data" 

# --- GCS Client Setup (IMPORTANT) ---
# You need to uncomment and use the actual library here.
try:
    from google.cloud import storage
    # Initialize the client. This typically handles authentication automatically
    # based on the environment (e.g., service account key or Colab/GCE context).
    GCS_CLIENT = storage.Client()
except ImportError:
    print("ERROR: 'google-cloud-storage' library not found. Please install it.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to initialize Google Cloud Storage client: {e}")
    sys.exit(1)


def load_npz_from_gcs(core_id: int, filename: str) -> dict:
    """
    Constructs the GCS path for a sharded NPZ file, downloads its content
    to a memory buffer, and loads the NumPy arrays from the buffer.

    Args:
        core_id: The TPU core index (0-31) which maps to the folder name (e.g., core_10).
        filename: The name of the NPZ file (e.g., embeddings_chunk_0.npz).

    Returns:
        A dictionary containing the NumPy arrays from the NPZ file, or None on failure.
    """
    # 1. Construct the full GCS Blob Path
    blob_name = f"{GCS_BASE_PREFIX}/core_{core_id}/{filename}"
    
    print("-" * 60)
    print(f"Core ID: {core_id}")
    print(f"Target Bucket: {BUCKET_NAME}")
    print(f"Target Blob:   {blob_name}")
    print("-" * 60)

    try:
        # 2. Get the Blob and Download to Memory
        bucket = GCS_CLIENT.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            print(f"❌ ERROR: Blob not found at gs://{BUCKET_NAME}/{blob_name}")
            return None

        # Create an in-memory binary stream (BytesIO)
        # This prevents writing the large file to disk.
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0) # Rewind the buffer to the beginning for NumPy to read it
        
        print("✅ GCS Download successful. Data loaded into memory buffer.")

        # 3. Load NPZ data from the memory buffer
        # The 'allow_pickle=False' is a security precaution for loading external data.
        # The data is expected to be {'all_layer_cls_tokens': ..., 'classifications': ...}
        npz_data = np.load(buffer, allow_pickle=False)

        # 4. Inspect and return the content
        
        # Keys check
        print(f"\n--- NPZ Content Check (Keys) ---")
        print(f"Keys found: {list(npz_data.keys())}")
        
        # CLS Tokens data
        cls_tokens = npz_data['all_layer_cls_tokens']
        print(f"\n'all_layer_cls_tokens' shape: {cls_tokens.shape}, Dtype: {cls_tokens.dtype}")
        
        # Classification data
        classifications = npz_data['classifications']
        print(f"'classifications' shape: {classifications.shape}, Dtype: {classifications.dtype}")
        
        if cls_tokens.shape[0] != classifications.shape[0]:
             print("❌ CRITICAL WARNING: Sample counts for CLS tokens and classifications do not match!")

        return dict(npz_data)

    except Exception as e:
        print(f"\nFATAL ERROR processing GCS or NPZ file: {e}")
        return None


if __name__ == '__main__':
    # --- Example Usage ---
    
    # Choose a specific core and chunk to test (based on your folder structure)
    TEST_CORE_ID = 10
    TEST_CHUNK_FILENAME = "embeddings_chunk_1.npz" # Use a small index for common chunks

    data = load_npz_from_gcs(TEST_CORE_ID, TEST_CHUNK_FILENAME)

    if data:
        print("\n--- Summary of Loaded Data ---")
        # Access the arrays by key
        cls_tokens_array = data['all_layer_cls_tokens']
        classifications_array = data['classifications']
        
        print(f"Total samples loaded: {cls_tokens_array.shape[0]}")
        
        # Display the first few classifications
        print(f"First 10 classification indices: {classifications_array[:10]}")
        
        # Display summary stats for the embedding vectors
        print(f"Mean of first embedding vector: {cls_tokens_array[0, 0, :].mean():.4f}")
    else:
        print("\nFailed to load data from GCS.")