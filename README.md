# SIGNLL — Early Exit Transformer Research

Research codebase for adaptive early exit in large language models, built on TPUv4 infrastructure with JAX/Flax and PyTorch/XLA.

## Project Overview

This project explores two complementary approaches to adaptive computation in LLMs:

1. **Regret-Aware Early Exit** (`src/llm_research/`) — A decision-theoretic framework that predicts downstream regret from hidden states to decide when early exit is safe. Uses counterfactual perturbation on code generation benchmarks (HumanEval + MBPP) with Qwen2.5-Coder-14B.

2. **Controller-Based Halting** (`src/training_job/`) — A gated halting mechanism using Conv1d entropy gates and SwiGLU transformers, trained on sentiment classification with SAM optimization.

## Repository Structure

```
src/
├── llm_research/       # Regret-aware early exit pipeline (Phase 2 research)
├── training_job/       # Controller model training (Phase 1)
├── tpu_job/            # Distributed inference & hidden state extraction
├── start_scripts/      # TPU pod provisioning & deployment
├── gcs/                # Google Cloud Storage utilities
├── final_tests/        # End-to-end integration tests
├── gemma_test/         # Gemma model TPU testing
├── text_encoder_tests/ # Text encoder validation (local + TPU)
├── torch_tests/        # PyTorch/XLA device checks
├── tpu_parallel_tests/ # Multi-host TPU verification
└── jax_tests/          # JAX framework testing

training_data/          # Local datasets (sentiment140)
```

## Infrastructure

- **Hardware**: TPUv4-64 pod (32 chips, 8 workers × 4 chips)
- **GCP Project**: `early-exit-transformer-network`
- **Zone**: `us-central2-b`
- **Frameworks**: JAX/Flax, EasyDeL, PyTorch/XLA, Transformers

## Quick Start

### Provision TPU Pod
```bash
bash src/start_scripts/gemma3_start_slice.sh
```

### Run Regret Training Pipeline
```bash
gcloud compute tpus tpu-vm ssh node-5 --worker=all \
  --command="cd ~/SIGNLL && source ~/edel_env/bin/activate && \
    PJRT_DEVICE=TPU python3 src/llm_research/regret_training.py"
```

### Run Controller Training
```bash
gcloud compute tpus tpu-vm ssh node-5 --worker=all \
  --command="cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/training_job/train4.py"
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| jax[tpu] | 0.4.33+ | TPU acceleration |
| flax | 0.11+ | Neural network modules (NNX API) |
| easydel | 0.2.0.2 | Multi-host LLM serving |
| torch | ~2.6.0 | Controller model training |
| torch_xla | ~2.6.0 | PyTorch TPU backend |
| transformers | latest | Tokenizers & model loading |
| datasets | latest | HumanEval/MBPP benchmarks |
| optax | latest | JAX optimizers |

## License

Apache License 2.0
