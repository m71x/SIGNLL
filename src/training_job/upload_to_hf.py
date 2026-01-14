import gcsfs
import numpy as np
import os
import shutil
from datasets import Dataset, Features, Array2D, ClassLabel, Value, load_dataset

# --- CONFIGURATION ---
GCS_PROJECT_ID = 'early-exit-transformer-network'
BUCKET_PATH = "encoder-models-2/siebert-data/siebert-actual-data/core_0"
HF_REPO_ID = "mxi71/twitter-100m-siebert-activations"
BATCH_SIZE_FILES = 5  # Number of NPZ files to process before uploading and clearing disk
CACHE_DIR = "/tmp/hf_cache"
# ---------------------

os.makedirs(CACHE_DIR, exist_ok=True)
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT_ID)
file_paths = sorted(fs.glob(f"{BUCKET_PATH}/*.npz"))
print(f"Found {len(file_paths)} files. Processing in batches of {BATCH_SIZE_FILES}...")

# 1. Define Features (Must stay consistent across all shards)
# Based on your logs: (25, 1024)
features = Features({
    "cls_tokens": Array2D(shape=(25, 1024), dtype="float32"),
    "label": ClassLabel(num_classes=2, names=["0", "1"]),
    "origin_file": Value("string")
})

def get_data_generator(files_subset):
    """Generator for a specific subset of files."""
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

# 2. Process in Shards
for i in range(0, len(file_paths), BATCH_SIZE_FILES):
    subset = file_paths[i : i + BATCH_SIZE_FILES]
    print(f"\n--- Processing batch {i//BATCH_SIZE_FILES + 1}: {len(subset)} files ---")
    
    # Create dataset for this shard
    shard_ds = Dataset.from_generator(
        get_data_generator(subset), 
        features=features,
        cache_dir=CACHE_DIR
    )
    
    # Push to Hub
    # If it's the first batch, we overwrite. Afterward, we append.
    if i == 0:
        shard_ds.push_to_hub(HF_REPO_ID, split="train")
        print("Initial shard uploaded.")
    else:
        shard_ds.push_to_hub(HF_REPO_ID, split="train")
        print(f"Shard appended. Total files processed: {i + len(subset)}")

    # 3. CRITICAL: Clear Disk Cache
    # We delete the cache folder and recreate it to ensure Errno 28 doesn't return
    shard_ds.cleanup_cache_files()
    shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)

print("\nAll shards uploaded successfully!")