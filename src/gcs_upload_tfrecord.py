import os
from google.cloud import storage

# === CONFIGURATION ===
# Replace with your GCS bucket name (no "gs://")
BUCKET_NAME = "your-tpu-bucket-name"  # e.g. "my-tpu-bucket"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training_data",
    "twitter_sentiment"
)
GCS_PATH = "datasets/twitter_sentiment"  # Path inside bucket

# === AUTHENTICATION ===
# Make sure you've set GOOGLE_APPLICATION_CREDENTIALS env var:
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"

def upload_directory_to_gcs(local_dir, bucket_name, gcs_path_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    total_files = 0
    uploaded_files = 0

    print(f"\n📤 Uploading files from: {local_dir}")
    print(f"→ Destination: gs://{bucket_name}/{gcs_path_prefix}/\n")

    for root, _, files in os.walk(local_dir):
        for filename in files:
            total_files += 1
            local_path = os.path.join(root, filename)
            # Preserve subdirectory structure relative to LOCAL_DIR
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_path_prefix, relative_path).replace("\\", "/")

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            uploaded_files += 1
            print(f"✅ Uploaded: {gcs_path}")

    print(f"\n🎉 Upload complete! {uploaded_files}/{total_files} files uploaded.")
    print(f"All files available under: gs://{bucket_name}/{gcs_path_prefix}/")


if __name__ == "__main__":
    if not os.path.exists(LOCAL_DIR):
        raise FileNotFoundError(f"❌ Local directory not found: {LOCAL_DIR}")

    upload_directory_to_gcs(LOCAL_DIR, BUCKET_NAME, GCS_PATH)
