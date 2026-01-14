import gcsfs
import numpy as np
import os
import shutil
from datasets import Dataset, Features, Array2D, ClassLabel, Value

# --- CONFIGURATION ---
GCS_PROJECT_ID = 'early-exit-transformer-network'
BUCKET_PATH = "encoder-models-2/siebert-data/siebert-actual-data/core_0"
HF_REPO_ID = "mxi71/twitter-100m-siebert-activations"
BATCH_SIZE_FILES = 5  
CACHE_DIR = "/tmp/hf_cache"
# ---------------------

os.makedirs(CACHE_DIR, exist_ok=True)
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT_ID)
file_paths = sorted(fs.glob(f"{BUCKET_PATH}/*.npz"))
print(f"Found {len(file_paths)} files. Processing in batches of {BATCH_SIZE_FILES}...")

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

# 2. Process in Shards
shard_counter = 0

for i in range(0, len(file_paths), BATCH_SIZE_FILES):
    subset = file_paths[i : i + BATCH_SIZE_FILES]
    print(f"\n--- Processing batch {shard_counter}: {len(subset)} files ---")
    
    # Create dataset for this shard
    shard_ds = Dataset.from_generator(
        get_data_generator(subset), 
        features=features,
        cache_dir=CACHE_DIR
    )
    
    # --- THE FIX: Unique Filenames ---
    # We create a unique filename for this shard.
    # Hugging Face Auto-Discovery will combine all "data/*.parquet" files into one "train" split.
    filename = f"data/train-{shard_counter:04d}.parquet"
    
    print(f"Pushing shard {shard_counter} as {filename}...")
    
    # We push the SHARD as a specific parquet file.
    # This prevents overwriting because every batch has a unique name.
    shard_ds.push_to_hub(HF_REPO_ID, path_in_repo=filename)
    
    shard_counter += 1

    # 3. Clean up
    shard_ds.cleanup_cache_files()
    shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)

print("\nAll shards uploaded successfully!")