import os
import json
import pandas as pd
import requests
import gcld3
from google.cloud import storage
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# --- Configuration ---
BUCKET_NAME = "encoder-models-2"
GCS_DATA_PREFIX = "twitter-100m"
GCS_CHECKPOINT_PREFIX = "tensorcore-checkpoints-v2-init"
CHECKPOINT_FILENAME = "progress.json"

# Assuming the data file format is tweets-XX-of-64.parquet
PARQUET_FILE_FORMAT = "tweets-{shard_index:02d}-of-64.parquet"

# --- GCS Helper Functions ---

def get_gcs_client_and_blob(bucket_name, gcs_blob_path):
    """Initializes GCS client, bucket, and blob object."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    return blob


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


# --- Helper: Checkpoint Reading ---

def read_checkpoint_index(core_id, local_dir):
    """Reads 'samples_seen' from checkpoint JSON and returns it as int."""
    gcs_blob_path = os.path.join(GCS_CHECKPOINT_PREFIX, f"core_{core_id}", CHECKPOINT_FILENAME)
    local_ckpt_path = os.path.join(local_dir, CHECKPOINT_FILENAME)

    print(f"[Core {core_id}] Reading checkpoint for row index: gs://{BUCKET_NAME}/{gcs_blob_path}", flush=True)

    if not download_file_from_gcs(local_ckpt_path, BUCKET_NAME, gcs_blob_path):
        print(f"[Core {core_id}] ❌ CHECKPOINT READ FAILED: file not found or failed to download.", flush=True)
        return None

    try:
        with open(local_ckpt_path, "r") as f:
            progress = json.load(f)
            index = progress.get("samples_seen")
            if index is not None and isinstance(index, int):
                print(f"[Core {core_id}] ✅ Checkpoint read successful: {index}", flush=True)
                return index
            else:
                print(f"[Core {core_id}] ❌ 'samples_seen' missing or invalid.", flush=True)
                return None
    except Exception as e:
        print(f"[Core {core_id}] ❌ Error parsing checkpoint JSON: {e}", flush=True)
        return None
    finally:
        if os.path.exists(local_ckpt_path):
            os.remove(local_ckpt_path)


# --- Distributed Parquet Loading Function ---

def run_distributed_parquet_load(core_id, total_cores, local_dir):
    """
    Reads the checkpoint index, downloads the corresponding Parquet shard,
    accesses the specific row, and classifies tweet language with gcld3.
    """
    # 1. READ CHECKPOINT INDEX
    row_index_base = read_checkpoint_index(core_id, local_dir)
    if row_index_base is None:
        return

    # Apply offset (matching user’s prior implementation)
    row_index = row_index_base + 8

    # 2. CONSTRUCT PARQUET PATHS
    shard_index = core_id
    parquet_filename = PARQUET_FILE_FORMAT.format(shard_index=shard_index)
    gcs_blob_path = os.path.join(GCS_DATA_PREFIX, parquet_filename)
    local_parquet_path = os.path.join(local_dir, parquet_filename)

    print(f"\n{'='*50}\n[Core {core_id}] Loading shard: {parquet_filename} (Row {row_index})\n{'='*50}", flush=True)
    print(f"[Core {core_id}] Target: gs://{BUCKET_NAME}/{gcs_blob_path}", flush=True)

    # 3. DOWNLOAD PARQUET FILE
    if not download_file_from_gcs(local_parquet_path, BUCKET_NAME, gcs_blob_path):
        print(f"[Core {core_id}] ❌ Parquet download failed.", flush=True)
        return

    # 4. READ & PROCESS
    try:
        df = pd.read_parquet(local_parquet_path, columns=['tweet'], engine='pyarrow')
        total_rows = len(df)

        if row_index < total_rows:
            tweet_text = str(df.iloc[row_index]['tweet']).replace('\n', ' ')

            # Initialize gcld3 detector
            detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
            result = detector.FindLanguage(text=tweet_text)

            if not result.is_reliable:
                print(f"[Core {core_id}] ⚠️ Language detection not reliable.", flush=True)

            is_english = result.language == 'en'
            confidence = result.probability

            print(f"[Core {core_id}] ✅ DATA ACCESS SUCCESS.", flush=True)
            print(f"[Core {core_id}] |-> Shard Rows: {total_rows}", flush=True)
            print(f"[Core {core_id}] |-> Language: {result.language} (Conf: {confidence:.4f})", flush=True)
            print(f"[Core {core_id}] |-> IS ENGLISH: {is_english}", flush=True)
            print(f"[Core {core_id}] |-> Reliable: {result.is_reliable}", flush=True)
            print(f"[Core {core_id}] |-> Tweet Content:\n{tweet_text}\n", flush=True)
        else:
            print(f"[Core {core_id}] ❌ Index {row_index} out of bounds (max {total_rows - 1}).", flush=True)

    except Exception as e:
        print(f"[Core {core_id}] ❌ DATA LOAD FAILED: {e}", flush=True)
    finally:
        if os.path.exists(local_parquet_path):
            os.remove(local_parquet_path)
            print(f"[Core {core_id}] Cleaned up local file: {local_parquet_path}", flush=True)


# --- Worker Function ---

def _mp_fn(index):
    core_id = xm.get_ordinal()
    total_cores = xm.xrt_world_size()

    local_dir = f"/tmp/tpu_parquet_load_{core_id}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[Core {core_id}] Using device {xm.xla_device()} / Total cores: {total_cores}", flush=True)

    # No model download needed for gcld3 (it's built-in)
    run_distributed_parquet_load(core_id, total_cores, local_dir)

    xm.rendezvous("parquet_load_complete")
    print(f"[Core {core_id}] ALL PHASES COMPLETE.", flush=True)


# --- Entry point ---
if __name__ == "__main__":
    # Ensure dependencies: pip install pandas pyarrow google-cloud-storage gcld3 requests
    xmp.spawn(_mp_fn, args=())
