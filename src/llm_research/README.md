# LLM Research — Regret-Aware Early Exit

Implementation of "Regret-Aware Early Exit for Large Language Models via Counterfactual Difficulty Estimation."

## Core Idea

Standard early exit methods (entropy, confidence) fail on hierarchical reasoning because they ignore the **downstream consequences** of exiting early. This framework reframes early exit as an **optimal stopping problem under risk**, where a lightweight MLP predicts the expected regret of committing to a partial computation.

## Pipeline (3 Phases)

### Phase 1: Baseline Generation
- Run Qwen2.5-Coder-14B with greedy decoding on ~600 code problems
- Execute generated code against test suites → binary reward R* (pass/fail)
- Output: `regret_baseline_results.json`

### Phase 2: Counterfactual Perturbation
- For each passing baseline, run a full forward pass to extract hidden states at target layers [4, 10, 16, 22, 28, 34]
- At each (layer, position), apply **directional logit perturbation**: decrease the chosen token's logit, boost top-k competitors
- Greedy rollout from the perturbed token → execute → get perturbed reward
- Regret = max(0, R* − R_perturbed)
- Output: `regret_dataset.npz` (hidden states + layer/position indices + regret values)

### Phase 3: MLP Training
- Train a small Flax MLP on the regret dataset
- Architecture: (hidden_dim+2) → 256 → GELU → 64 → GELU → 1 (ReLU)
- Input: hidden state vector + normalized layer index + normalized position
- Output: predicted regret (scalar ≥ 0)
- Output: `regret_estimator_weights.npz`

## Files

| File | Purpose |
|------|---------|
| `regret_training.py` | Main 3-phase pipeline (runs on 32 TPUv4s via EasyDeL) |
| `regret_estimator.py` | Flax NNX MLP model for regret prediction |
| `code_executor.py` | Sandboxed Python code runner with timeout |
| `code_prompts.py` | HumanEval + MBPP dataset loader (~600 problems) |
| `Phase_2_Research_Plan.md` | Full research specification |
| `elarge_test.py` | Multi-host generation + activation extraction test |
| `activation_patch_test.py` | Multi-host activation perturbation test |
| `qwen_test.py` | Basic multi-prompt generation test |
| `easydel_test.py` | EasyDeL initialization test |

## Running

```bash
# From repo root on TPU pod (all workers)
cd ~/SIGNLL && source ~/edel_env/bin/activate
pip install datasets  # first time only
PJRT_DEVICE=TPU python3 src/llm_research/regret_training.py
```

Or via tmux for long-running jobs:
```bash
gcloud compute tpus tpu-vm ssh node-5 --worker=all \
  --command="cd ~/SIGNLL && source ~/edel_env/bin/activate && \
    tmux new -d -s regret 'PJRT_DEVICE=TPU python3 src/llm_research/regret_training.py'"
```

## Configuration

Key parameters in `regret_training.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_ID` | Qwen2.5-Coder-14B-Instruct | Base model |
| `MAX_GEN_TOKENS` | 512 | Max generation length |
| `TARGET_LAYERS` | [4, 10, 16, 22, 28, 34] | Layers to probe (~6 of 48) |
| `PERTURB_STRENGTH` | 2.0 | Logit perturbation magnitude (in σ) |
| `PERTURB_TEMP` | 0.3 | Sampling temperature for perturbed tokens |
| `PERTURB_TOP_K` | 3 | Number of competitor tokens |
| `MAX_POSITIONS_PER_PROMPT` | 20 | Token positions to perturb per layer |
| `ESTIMATOR_LR` | 1e-4 | MLP learning rate |
| `ESTIMATOR_EPOCHS` | 50 | MLP training epochs |

## Multi-Host Patterns

- `jax.distributed.initialize()` must come before EasyDeL imports
- Master (process 0) handles I/O, code execution, and file saves
- `multihost_utils.sync_global_devices()` for barrier synchronization
- `multihost_utils.process_allgather()` to broadcast rewards across workers
- EasyDeL output validation is monkey-patched for multi-host eager mode
- Sharding: axis_dims=(1, 1, 8, 4, 1) for dp/fsdp/tp/sp/selective
