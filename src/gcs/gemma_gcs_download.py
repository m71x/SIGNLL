from huggingface_hub import snapshot_download
import subprocess

# Hugging Face model repo
repo_id = "siebert/sentiment-roberta-large-english"

# Local temporary folder
local_dir = "/tmp/siebert"

# Download the full model snapshot locally (all files)
snapshot_download(repo_id, cache_dir=local_dir, local_dir=local_dir)

# Copy to GCS bucket
gcs_bucket_path = "gs://encoder-models/siebert"
subprocess.run(f"gsutil -m cp -r {local_dir}/* {gcs_bucket_path}/", shell=True)
