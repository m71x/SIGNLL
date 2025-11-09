import os
from datasets import load_dataset
from google.cloud import storage

BUCKET_NAME = "encoder-models"              # <-- your bucket
GCS_PREFIX = "twitter-100m"          # <-- folder path inside bucket
LOCAL_DIR = "/home/mikexi/twitter100m_shards"
NUM_SHARDS = 64                             # match your TPU core count

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Loading Hugging Face dataset 'enryu43/twitter100m_tweets' ...", flush=True)
dataset = load_dataset("enryu43/twitter100m_tweets", split="train")

total = len(dataset)
shard_size = total // NUM_SHARDS
print(f"Total samples: {total}, shard size ≈ {shard_size}")

# --- Shard locally ---
for i in range(NUM_SHARDS):
    start = i * shard_size
    end = total if i == NUM_SHARDS - 1 else (i + 1) * shard_size
    shard = dataset.select(range(start, end))
    shard_path = os.path.join(LOCAL_DIR, f"tweets-{i:02d}-of-{NUM_SHARDS:02d}.parquet")
    print(f"[Shard {i}] Saving {end-start} rows → {shard_path}")
    shard.to_parquet(shard_path)

print("✅ All shards saved locally. Uploading to GCS...", flush=True)

# --- Upload to GCS ---
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

for filename in os.listdir(LOCAL_DIR):
    local_path = os.path.join(LOCAL_DIR, filename)
    gcs_path = f"{GCS_PREFIX}/{filename}"
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → gs://{BUCKET_NAME}/{gcs_path}", flush=True)

print("✅ All shards uploaded successfully!")
