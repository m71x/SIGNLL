import gcsfs
import numpy as np
import os
import shutil
from datasets import Dataset, Features, Array2D, ClassLabel, Value
from huggingface_hub import HfApi # Import the API helper

# --- CONFIGURATION ---
GCS_PROJECT_ID = 'early-exit-transformer-network'
BUCKET_PATH = "encoder-models-2/siebert-data/siebert-actual-data/core_0"
HF_REPO_ID = "mxi71/twitter-100m-siebert-activations"
BATCH_SIZE_FILES = 5  
CACHE_DIR = "/tmp/hf_cache"
PARQUET_DIR = "/tmp/hf_parquet" # Temp folder for parquet files
# ---------------------

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

fs = gcsfs.GCSFileSystem(project=GCS_PROJECT_ID)
file_paths = sorted(fs.glob(f"{BUCKET_PATH}/*.npz"))
print(f"Found {len(file_paths)} files. Processing in batches of {BATCH_SIZE_FILES}...")

# Initialize API
api = HfApi()

features = Features({
    "cls_tokens": Array2D(shape=(25, 1024), dtype="float32"),
    "label": ClassLabel(num_classes=2, names=["0", "1"]),
    "origin_file": Value("string")
})

def get_data_generator(files_subset):
    def gen():
        for file_path in files_subset:
            with fs.open(file_path, 'rb') as f:
                data = np.load(f)
                cls_tokens = data['all_layer_cls_tokens']
                labels = data['classifications']
                for i in range(len(cls_tokens)):
                    yield {
                        "cls_tokens": cls_tokens[i],
                        "label": int(labels[i]),
                        "origin_file": file_path.split("/")[-1]
                    }
    return gen

shard_counter = 0

for i in range(0, len(file_paths), BATCH_SIZE_FILES):
    subset = file_paths[i : i + BATCH_SIZE_FILES]
    print(f"\n--- Processing batch {shard_counter}: {len(subset)} files ---")
    
    # 1. Create dataset object
    shard_ds = Dataset.from_generator(
        get_data_generator(subset), 
        features=features,
        cache_dir=CACHE_DIR
    )
    
    # 2. Save as Local Parquet
    local_parquet_path = os.path.join(PARQUET_DIR, f"train-{shard_counter:04d}.parquet")
    print(f"Saving locally to {local_parquet_path}...")
    shard_ds.to_parquet(local_parquet_path)
    
    # 3. Upload File to Hub (This supports path_in_repo)
    repo_path = f"data/train-{shard_counter:04d}.parquet"
    print(f"Uploading to {HF_REPO_ID} as {repo_path}...")
    
    api.upload_file(
        path_or_fileobj=local_parquet_path,
        path_in_repo=repo_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    
    shard_counter += 1

    # 4. Cleanup
    shard_ds.cleanup_cache_files()
    shutil.rmtree(CACHE_DIR)
    shutil.rmtree(PARQUET_DIR) # Clear parquet file to save space
    os.makedirs(CACHE_DIR)
    os.makedirs(PARQUET_DIR)

print("\nAll shards uploaded successfully!")