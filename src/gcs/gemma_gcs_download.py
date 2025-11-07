from huggingface_hub import snapshot_download
import subprocess

# Hugging Face model repo
repo_id = "google/gemma-3-4b-it"

# Local temporary folder
local_dir = "/tmp/gemma-3-4b-it"

# Download the full model snapshot locally (all files)
snapshot_download(repo_id, cache_dir=local_dir, local_dir=local_dir)

# Copy to GCS bucket
gcs_bucket_path = "gs://startup-scripts-gemma3/gemma3-4b/gemma-3-4b-it"
subprocess.run(f"gsutil -m cp -r {local_dir}/* {gcs_bucket_path}/", shell=True)
