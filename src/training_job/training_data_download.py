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
try:
    from google.cloud import storage
    # Initialize the client. This typically handles authentication automatically
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
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0) # Rewind the buffer to the beginning for NumPy to read it
        
        # --- CRITICAL CHECK: Buffer size and header ---
        buffer_size = buffer.getbuffer().nbytes
        print(f"✅ GCS Download successful. Data loaded into memory buffer ({buffer_size} bytes).")

        if buffer_size == 0:
            print(f"❌ CRITICAL ERROR: Downloaded file is EMPTY (0 bytes).")
            return None
        
        if buffer_size < 1024:
             print(f"⚠️ WARNING: Downloaded file size is suspiciously small.")


        # 3. Load NPZ data from the memory buffer
        npz_data = np.load(buffer, allow_pickle=False)

        # 4. Inspect and return the content
        print(f"\n--- NPZ Content Check (Keys) ---")
        print(f"Keys found: {list(npz_data.keys())}")
        
        cls_tokens = npz_data['all_layer_cls_tokens']
        classifications = npz_data['classifications']
        
        print(f"\n'all_layer_cls_tokens' shape: {cls_tokens.shape}, Dtype: {cls_tokens.dtype}")
        print(f"'classifications' shape: {classifications.shape}, Dtype: {classifications.dtype}")
        
        if cls_tokens.shape[0] != classifications.shape[0]:
             print("❌ CRITICAL WARNING: Sample counts for CLS tokens and classifications do not match!")

        return dict(npz_data)

    except KeyError as e:
        print(f"\nFATAL ERROR: Missing expected key {e} in NPZ file.")
        return None
    except OSError as e:
        print(f"\nFATAL ERROR processing NPZ file: {e}")
        return None
    except Exception as e:
        print(f"\nFATAL ERROR during GCS communication or other unexpected issue: {e}")
        return None

# --- NEW FUNCTION FOR TRAINING DATA PREP ---

def training_data_download(core_id: int, filename: str, max_entries: int) -> dict:
    """
    Downloads NPZ data from GCS, selects the first N entries (for speed), 
    and then shuffles the data before returning it.

    Args:
        core_id: The TPU core index.
        filename: The name of the NPZ file.
        max_entries: The maximum number of entries to return.

    Returns:
        A dictionary containing the shuffled and sliced NumPy arrays, or None on failure.
    """
    # 1. Load the data
    data = load_npz_from_gcs(core_id, filename)
    if data is None:
        return None

    cls_tokens = data['all_layer_cls_tokens']
    classifications = data['classifications']
    
    total_samples = cls_tokens.shape[0]

    # 2. Determine the slice size for speed/testing
    N = min(total_samples, max_entries)
    
    print(f"\n--- Data Preparation ---")
    print(f"Total samples found: {total_samples}")
    print(f"Samples selected (N): {N} (max_entries={max_entries})")

    if N == 0:
        print("❌ Data slice resulted in 0 samples.")
        return None
    
    # 3. Create a shuffle index (0 to N-1)
    # We only create indices up to N, which performs the slice implicitly
    indices = np.arange(total_samples)
    
    # Create a random permutation of all available indices
    np.random.shuffle(indices)
    
    # Select the first N shuffled indices to achieve both shuffle and slice
    # This is more efficient than slicing first, then shuffling.
    shuffled_and_sliced_indices = indices[:N]

    # 4. Apply the shuffle and slice to the arrays
    shuffled_cls_tokens = cls_tokens[shuffled_and_sliced_indices]
    shuffled_classifications = classifications[shuffled_and_sliced_indices]
    
    print(f"✅ Data successfully shuffled and sliced. Final samples: {shuffled_cls_tokens.shape[0]}")
    
    # 5. Return the shuffled data
    return {
        'all_layer_cls_tokens': shuffled_cls_tokens,
        'classifications': shuffled_classifications
    }


if __name__ == '__main__':
    # --- Example Usage for Training Data ---
    
    # Configuration to test
    TEST_CORE_ID = 10
    TEST_CHUNK_FILENAME = "embeddings_chunk_1.npz"
    # User-defined limit for the number of entries
    SAMPLES_TO_LOAD = 19500 

    # Call the new function
    training_data = training_data_download(
        core_id=TEST_CORE_ID, 
        filename=TEST_CHUNK_FILENAME, 
        max_entries=SAMPLES_TO_LOAD
    )

    if training_data:
        print("\n--- Summary of Prepared Training Data ---")
        
        # Access the arrays by key
        cls_tokens_array = training_data['all_layer_cls_tokens']
        classifications_array = training_data['classifications']
        
        print(f"Total samples returned: {cls_tokens_array.shape[0]}")
        
        # Check that the data is shuffled (first 10 classifications should be random)
        print(f"First 10 classification indices (Shuffled): {classifications_array[:10]}")
        
        # Display summary stats for the embedding vectors
        print(f"Mean of first embedding vector: {cls_tokens_array[0, 0, :].mean():.4f}")
    else:
        print("\nFailed to load and prepare training data.")