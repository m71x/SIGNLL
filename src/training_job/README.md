# Training Job — Controller Model for Adaptive Early Exit

Gated halting controller that learns which transformer layers to exit at, trained on sentiment classification with distributed PyTorch/XLA on TPU.

## Architecture

### Controller Model (`controller_model.py`)
- **RoPE** — Rotary position embeddings for sequence-aware attention
- **SwiGLU FFN** — Gated linear unit with SiLU activation (d_model → 2×hidden → d_model)
- **CustomTransformerLayer** — Pre-norm MHA + SwiGLU + residual connections
- **VectorizedEntropyGate** — Conv1d-based parallel gating across L layers
  - Input: (batch, d_ctrl × L) → 8×L channels → 1×L gate values
  - Sigmoid output ∈ [0, 1] per layer (exit probability)

### GRU Variant (`controller_gru.py`)
- Alternative controller using GRU recurrence instead of attention
- Same gating mechanism for exit decisions

## Training

### Optimizer: SAM (Sharpness-Aware Minimization)
- Two-step process: perturb weights to local max, then update
- Encourages flatter loss landscapes for better generalization
- Default perturbation: ρ = 0.05

### Scheduler
- `CosineAnnealingWarmRestarts` for learning rate cycling

### Data
- Sentiment140 dataset (1.6M Twitter samples, binary classification)
- Downloaded and converted to TFRecord format
- Distributed across TPU workers via `DistributedSampler`

## Files

| File | Purpose |
|------|---------|
| `controller_model.py` | Gated halting controller (attention-based) |
| `controller_gru.py` | GRU-based controller variant |
| `controller_utils.py` | Shared utilities |
| `train4.py` | Latest training script (SAM + cosine annealing) |
| `train.py` → `train3.py` | Earlier training iterations |
| `train_gru.py` | GRU variant training |
| `train_only.py` | Training without evaluation |
| `eval_only.py` / `eval_gru.py` | Standalone evaluation |
| `training_data_download.py` | Sentiment140 download + TFRecord conversion |
| `upload_to_hf.py` | Upload trained model to HuggingFace Hub |
| `test_stage2_*.py` | Debugging scripts (hangs, NaN, speed) |
| `improvements.txt` | Training notes (CLS loss plateau, halting diversity) |
| `notes.txt` | General development notes |

## Running

```bash
# Download training data (first time)
python3 src/training_job/training_data_download.py

# Train on TPU pod
gcloud compute tpus tpu-vm ssh node-5 --worker=all \
  --command="cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/training_job/train4.py"
```

## Known Issues

Documented in `improvements.txt`:
- CLS loss plateaus around 0.1–0.15
- Halting layer diversity collapses (all samples exit at same layer)
- Planned fixes: Ponder Loss regularization, Gumbel-Softmax for discrete gates, dual information objective
