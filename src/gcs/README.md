# GCS — Google Cloud Storage Utilities

Helpers for uploading/downloading models, datasets, and checkpoints to GCS.

## Files

| File | Purpose |
|------|---------|
| `gcs_access_test.py` | Core GCS upload/download with exponential backoff retry (max 3 attempts, 300s timeout) |
| `checkpoint_folder_creation.py` | Initialize checkpoint directory structure in GCS for distributed workers |
| `gcs_upload_tfrecord.py` | Upload TFRecord datasets preserving directory structure |
| `gemma_gcs_download.py` | Download Gemma model weights from GCS to local storage |
| `twitter_100m_upload_shard.py` | Shard and distribute Twitter dataset across GCS |
| `tfrecords_conversion.py` | Convert raw datasets to TFRecord format |

## Features

- Exponential backoff retry on upload/download failures
- File size and hash verification after transfer
- Local file existence checks before upload
- XLA core ID logging for tracking distributed worker I/O
- 300-second timeout per upload operation
