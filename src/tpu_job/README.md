# TPU Job — Distributed Inference & Hidden State Extraction

Multi-host inference pipeline that processes dataset shards through SentenceBERT on TPU, extracts CLS token hidden states at every layer, and uploads results to GCS.

## Pipeline

1. **Model Download** — Master (core 0) fetches SentenceBERT (Siebert) from GCS bucket `encoder-models-2/siebert`
2. **Pre-flight Checks** — Verify CLS token extraction (25 hidden states), classification output (binary), language detection (gcld3)
3. **Shard Processing** — Each TPU core loads its assigned parquet shard, runs batched inference (batch_size=64)
4. **Hidden State Extraction** — Extract CLS tokens at each of the 25 layers (embedding + 24 transformer layers)
5. **Accumulation & Upload** — Buffer 20,000 samples, then upload NPZ to GCS
6. **Checkpointing** — Track last processed sample index per worker for resume capability

## Files

| File | Purpose |
|------|---------|
| `main.py` / `main2.py` | Multi-host inference orchestrator |
| `inference.py` | Model inference + CLS token extraction |
| `gcs_access.py` | GCS upload/download with retry logic |
| `gcs_npz_loader.py` | NumPy dataset loading from GCS |
| `checkpoint_access.py` | GCS checkpoint save/load/resume |
| `training_dataset_access.py` | Dataset shard partitioning |
| `npz_file_validation.py` | Dataset integrity validation |
| `reset_checkpoints.py` | Reset all worker checkpoints to 0 |
| `tpu_core_count.py` | TPU device enumeration |
| `verify_tpu_cores.py` | TPU connectivity verification |

## Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `INFERENCE_BATCH_SIZE` | 64 | Samples per forward pass |
| `IO_ACCUMULATION_THRESHOLD` | 20,000 | Samples before GCS upload |
| `EXPECTED_CLS_COUNT` | 25 | Hidden layers + embedding |
| Checkpoint prefix | `tensorcore-checkpoints-v2-init` | GCS checkpoint path |

## Running

```bash
gcloud compute tpus tpu-vm ssh node-5 --worker=all \
  --command="cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/tpu_job/main2.py"
```
