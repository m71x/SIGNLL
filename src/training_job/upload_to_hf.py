import gcsfs
import numpy as np
import os
import shutil
from datasets import Dataset, Features, Array2D, ClassLabel, Value
from huggingface_hub import HfApi

# --- CONFIGURATION ---
GCS_PROJECT_ID = 'early-exit-transformer-network'
# Base path without the "core_X" suffix
BASE_BUCKET_PATH = "encoder-models-2/siebert-data/siebert-actual-data"
HF_REPO_ID = "mxi71/twitter-100m-siebert-activations"
BATCH_SIZE_FILES = 5  
CACHE_DIR = "/tmp/hf_cache"
PARQUET_DIR = "/tmp/hf_parquet"
# ---------------------

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

# Initialize API and GCS
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT_ID)
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

# --- MAIN LOOP FOR CORES 1 THROUGH 10 ---
for core_num in range(13, 19):
    current_bucket_path = f"{BASE_BUCKET_PATH}/core_{core_num}"
    print(f"\n\n=== STARTING CORE {core_num} ===")
    print(f"Reading from: {current_bucket_path}")
    
    file_paths = sorted(fs.glob(f"{current_bucket_path}/*.npz"))
    
    if not file_paths:
        print(f"No files found for core_{core_num}. Skipping.")
        continue
        
    print(f"Found {len(file_paths)} files. Processing in batches of {BATCH_SIZE_FILES}...")

    # Reset shard counter for this core
    shard_counter = 0

    for i in range(0, len(file_paths), BATCH_SIZE_FILES):
        subset = file_paths[i : i + BATCH_SIZE_FILES]
        print(f"\n[Core {core_num}] Processing batch {shard_counter}: {len(subset)} files ---")
        
        # 1. Create dataset object
        shard_ds = Dataset.from_generator(
            get_data_generator(subset), 
            features=features,
            cache_dir=CACHE_DIR
        )
        
        # 2. Save as Local Parquet
        # We include the core number in the filename to avoid overwriting other cores
        local_filename = f"core_{core_num}_train-{shard_counter:04d}.parquet"
        local_parquet_path = os.path.join(PARQUET_DIR, local_filename)
        
        print(f"Saving locally to {local_parquet_path}...")
        shard_ds.to_parquet(local_parquet_path)
        
        # 3. Upload File to Hub
        # The filename in the repo will also include "core_X"
        repo_path = f"data/{local_filename}"
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
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        if os.path.exists(PARQUET_DIR):
            shutil.rmtree(PARQUET_DIR) 
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(PARQUET_DIR, exist_ok=True)

    print(f"=== FINISHED CORE {core_num} ===")

print("\nAll cores uploaded successfully!")