# Final Tests — End-to-End Integration Tests

Integration tests validating the full distributed pipeline: data access, inference, GCS I/O, and checkpointing.

## Tests

| Test | Validates |
|------|-----------|
| `checkpoint_access_test.py` | GCS checkpoint save/load round-trip |
| `checkpoint_access_test2.py` | Extended checkpoint scenarios |
| `checkpoint_access_only_test.py` | Checkpoint recovery on worker restart |
| `gcs_access_dataset_test.py` | Dataset shard download from GCS |
| `hidden_layer_upload_test.py` | Upload 3D hidden state arrays (batch × layers × dims) |
| `npz_access_test.py` | Basic NPZ serialization/deserialization |
| `training_data_download_test.py` | Dataset shard partitioning and retrieval |
| `parallel_run_test.py` | Multi-process TPU execution |
| `set_checkpoints_0.py` | Reset all worker checkpoints to 0 |

## Running

```bash
# Single test on all workers
gcloud compute tpus tpu-vm ssh node-5 --worker=all \
  --command="cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/final_tests/checkpoint_access_test.py"
```
