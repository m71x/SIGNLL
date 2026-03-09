# Start Scripts — TPU Pod Provisioning & Deployment

Shell scripts for creating, configuring, and managing TPUv4 pod slices on GCP.

## Scripts

### `gemma3_start_slice.sh` (Primary)

Full lifecycle management for a TPUv4-64 pod:

| Command | Description |
|---------|-------------|
| `create` | Provision TPU VM (`v4-64`, `us-central2-b`, `tpu-ubuntu2204-base`) |
| `install` | Install PyTorch 2.6 + XLA, EasyDeL, datasets, gcld3, etc. |
| `clone` | Clone repo to all workers (`--worker=all`) |
| `run` | Launch training via tmux on all workers |
| `run_elarge` | Run EasyDeL large model tests (with `edel_env` venv) |
| `kill` | Kill Python processes on all workers |
| `monitor` | View logs (`syslog`, `journalctl`, `htop`, `vmstat`) |
| `cache` | Flush HuggingFace cache and temp files |
| `git_sync` | `git fetch && git reset --hard origin/main` on all workers |
| `delete` | Destroy TPU VM |

### `gemma3_start.sh`
Simpler variant focused on Gemma model setup.

## GCP Configuration

```
Project:      early-exit-transformer-network
TPU Name:     node-5
Zone:         us-central2-b
Accelerator:  v4-64 (32 TPUv4 chips, 8 workers × 4 chips)
Runtime:      tpu-ubuntu2204-base
```

## Usage

```bash
# Create and setup TPU pod
bash src/start_scripts/gemma3_start_slice.sh create
bash src/start_scripts/gemma3_start_slice.sh install
bash src/start_scripts/gemma3_start_slice.sh clone

# Deploy code updates
bash src/start_scripts/gemma3_start_slice.sh git_sync

# Run a job
bash src/start_scripts/gemma3_start_slice.sh run

# Teardown
bash src/start_scripts/gemma3_start_slice.sh kill
bash src/start_scripts/gemma3_start_slice.sh delete
```

## Environment Notes

- EasyDeL jobs use a dedicated venv: `source ~/edel_env/bin/activate`
- Shared memory may need remounting: `sudo mount -o remount,size=16G /dev/shm`
- All `--worker=all` commands execute in parallel across the 8 host VMs
