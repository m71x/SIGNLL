import os
import json
from google.cloud import storage

# --- Configuration Constants ---
# NOTE: These constants define default paths and names but can be overridden 
# by the calling script if needed.
BUCKET_NAME = "encoder-models-2"
MODEL_PREFIX = "siebert"
LOCAL_MODEL_PATH = "/home/mikexi/siebert_model"
UPLOAD_PREFIX = "siebert-data/siebert-data-test"


# --- Helper: Upload file to GCS ---
def upload_file_to_gcs(bucket_name, local_path, gcs_blob_path):
    """
    Uploads a file from local storage to a GCS blob.
    
    Args:
        bucket_name (str): The name of the GCS bucket.
        local_path (str): The local path of the file to upload.
        gcs_blob_path (str): The destination path/name in the bucket.
        
    Returns:
        bool: True if upload was successful, False otherwise.
    """
    try:
        # Note: GCS client is initialized inside the function for independence
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_path)
        blob.upload_from_filename(local_path)
        print(f"✅ Uploaded: {local_path} → gs://{bucket_name}/{gcs_blob_path}", flush=True)
        return True
    except Exception as e:
        print(f"❌ Upload failed for {local_path}: {e}", flush=True)
        return False


# --- Helper: Download file from GCS ---
def download_file_from_gcs(bucket_name, gcs_blob_path, local_path):
    """
    Downloads a file from GCS to local storage.
    
    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_blob_path (str): The source path/name in the bucket.
        local_path (str): The destination local file path.
        
    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"✅ Downloaded: gs://{bucket_name}/{gcs_blob_path} → {local_path}", flush=True)
        return True
    except Exception as e:
        print(f"❌ Download failed for {gcs_blob_path}: {e}", flush=True)
        return False


# --- Model directory download (only by master) ---
def download_model_from_gcs(bucket_name, prefix, local_dir):
    """
    Recursively downloads all files under a given GCS prefix to a local directory.
    
    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The GCS prefix (directory) containing model files.
        local_dir (str): The local destination directory.
    """
    print(f"Downloading model from gs://{bucket_name}/{prefix} ...", flush=True)
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List all blobs under the given prefix
    blobs = bucket.list_blobs(prefix=prefix)
    count = 0
    for blob in blobs:
        # Skip GCS 'folders' (blobs ending with /)
        if blob.name.endswith("/"):
            continue
            
        # Determine relative path and local destination
        rel_path = blob.name[len(prefix):].lstrip("/")
        local_file = f"{local_dir}/{rel_path}"
        
        # Ensure parent directories exist locally
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)
        count += 1
        
    print(f"Model download complete ({count} files).", flush=True)