"""
Regret-Aware Early Exit Training Pipeline
===========================================
Three-phase pipeline:
  Phase 1: Greedy baseline generation on HumanEval/MBPP → reward R*
  Phase 2: Counterfactual perturbation → regret collection
  Phase 3: Train MLP regret estimator in JAX/Optax

Built on multi-host patterns from elarge_test.py / activation_patch_test.py.
Runs Qwen2.5-Coder-14B on 32 TPUv4s via EasyDeL.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import gc
import json
import inspect
import time
import os

# ── CONFIGURABLE PARAMETERS ──────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
MAX_GEN_TOKENS = 512         # Max tokens for code generation
TARGET_LAYERS = [4, 10, 16, 22, 28, 34]  # Sample ~6 layers out of 48
PERTURB_STRENGTH = 2.0       # Logit perturbation magnitude
PERTURB_TEMP = 0.3           # Temperature for perturbed sampling
PERTURB_TOP_K = 3            # Top-k competitors for perturbation
MAX_POSITIONS_PER_PROMPT = 20  # Cap token positions to perturb per layer
BATCH_SIZE = 4               # Prompts to process before GC
CODE_TIMEOUT = 10            # Seconds for code execution

# Regret estimator training
ESTIMATOR_LR = 1e-4
ESTIMATOR_EPOCHS = 50
ESTIMATOR_BATCH = 256
TRAIN_SPLIT = 0.8

# Output paths
BASELINE_PATH = "regret_baseline_results.json"
REGRET_DATA_PATH = "regret_dataset.npz"
ESTIMATOR_PATH = "regret_estimator_weights"

# ── 1. INITIALIZE DISTRIBUTED SYSTEM ─────────────────────────────────
jax.distributed.initialize()
from jax.experimental import multihost_utils
import easydel as ed
from transformers import AutoTokenizer

# Add script directory to path so local imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_prompts import load_all_problems, build_code_prompt
from code_executor import execute_code, extract_code_from_response

is_master = jax.process_index() == 0

if is_master:
    print(f"{'='*70}")
    print(f"{'REGRET-AWARE EARLY EXIT TRAINING':^70}")
    print(f"{'='*70}")
    print(f"  Devices:       {jax.device_count()}")
    print(f"  Local devices: {jax.local_device_count()}")
    print(f"  Target layers: {TARGET_LAYERS}")
    print(f"  Model:         {MODEL_ID}")
    print(f"{'='*70}")

# ── 2. LOAD DATASET ──────────────────────────────────────────────────
if is_master:
    print("\nLoading HumanEval + MBPP datasets...")

# All workers load the dataset (deterministic, no desync risk)
problems = load_all_problems()

if is_master:
    print(f"  Loaded {len(problems)} problems")

# ── 3. LOAD MODEL & TOKENIZER ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

axis_dims = (1, 1, 8, 4, 1)
axis_names = ("dp", "fsdp", "tp", "sp", "selective")

elm = (
    ed.eLargeModel.from_pretrained(MODEL_ID)
    .set_dtype("bf16")
    .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
)

# Only configure eSurge if we'll need it for Phase 1
# (skip if Phase 1 checkpoint is complete)
_phase1_done = False
if is_master and os.path.exists(BASELINE_PATH):
    try:
        with open(BASELINE_PATH) as f:
            _n = len(json.load(f))
        _phase1_done = _n >= len(problems)
    except Exception:
        pass

_p1_arr = jnp.array([1.0 if (is_master and _phase1_done) else 0.0])
_p1_arr = multihost_utils.process_allgather(_p1_arr)
_phase1_done = float(_p1_arr.flatten()[0]) > 0.5

if not _phase1_done:
    elm = elm.set_esurge(max_model_len=4096, max_num_seqs=4)
    if is_master:
        print("  eSurge configured for Phase 1 generation")
else:
    if is_master:
        print("  Phase 1 complete — skipping eSurge configuration")


# ═════════════════════════════════════════════════════════════════════
# PHASE 1: BASELINE GENERATION
# ═════════════════════════════════════════════════════════════════════

# Check if Phase 1 is already complete (checkpoint with all problems)
phase1_complete = False
baseline_results = []
if is_master and os.path.exists(BASELINE_PATH):
    try:
        with open(BASELINE_PATH) as f:
            baseline_results = json.load(f)
        if len(baseline_results) >= len(problems):
            phase1_complete = True
            print(f"\n  Phase 1 already complete: {len(baseline_results)} results loaded from checkpoint")
        else:
            print(f"\n  Partial checkpoint: {len(baseline_results)}/{len(problems)}")
    except Exception as e:
        print(f"  Could not load checkpoint: {e}, starting fresh")
        baseline_results = []

# Broadcast phase1_complete to all workers
p1_arr = jnp.array([1.0 if (is_master and phase1_complete) else 0.0])
p1_arr = multihost_utils.process_allgather(p1_arr)
phase1_complete = float(p1_arr.flatten()[0]) > 0.5

if not phase1_complete:
    if is_master:
        print(f"\n{'='*70}")
        print("PHASE 1: Generating baseline code with greedy decoding")
        print(f"{'='*70}")

    esurge = elm.build_esurge()

    # Determine start index from checkpoint
    start_idx = len(baseline_results)
    start_arr = jnp.array([start_idx if is_master else 0])
    start_arr = multihost_utils.process_allgather(start_arr)
    start_idx = int(start_arr.flatten()[0])

    for p_idx in range(start_idx, len(problems)):
        problem = problems[p_idx]
        conversation = build_code_prompt(problem)

        if is_master:
            print(f"\n[{p_idx+1}/{len(problems)}] {problem.task_id}")
            print(f"  Generating...", end="", flush=True)

        generated_text = ""
        num_tokens = 0
        t0 = time.time()

        # Greedy decode: temp=0 equivalent via SamplingParams
        for output in esurge.chat(
            conversation,
            sampling_params=ed.SamplingParams(
                max_tokens=MAX_GEN_TOKENS,
                temperature=0.0,
            ),
            stream=True,
        ):
            generated_text += output.delta_text
            num_tokens += 1

        elapsed = time.time() - t0

        # Extract code from response
        code = extract_code_from_response(generated_text)

        # Execute tests — only master runs code (CPU-bound)
        reward = 0.0
        exec_info = {}
        if is_master:
            reward, exec_info = execute_code(code, problem.test_code, timeout=CODE_TIMEOUT)
            print(f" {num_tokens} tok, {elapsed:.1f}s, R*={reward}")
            if reward == 0.0 and exec_info.get("stderr"):
                print(f"    Error: {exec_info['stderr'][:100]}")

        # Broadcast reward to all workers
        reward_arr = jnp.array([reward if is_master else 0.0])
        reward_arr = multihost_utils.process_allgather(reward_arr)
        reward = float(reward_arr.flatten()[0])  # Master's value at index 0

        baseline_results.append({
            "task_id": problem.task_id,
            "source": problem.source,
            "prompt_idx": p_idx,
            "generated_text": generated_text,
            "code": code,
            "reward": reward,
            "num_tokens": num_tokens,
            "elapsed": elapsed,
        })

        # Periodic GC and checkpoint saving
        if (p_idx + 1) % BATCH_SIZE == 0:
            gc.collect()
            if is_master and (p_idx + 1) % (BATCH_SIZE * 5) == 0:
                with open(BASELINE_PATH, "w") as f:
                    json.dump(baseline_results, f, indent=2, default=str)
                print(f"  [Checkpoint saved: {len(baseline_results)}/{len(problems)}]")
            multihost_utils.sync_global_devices(f"baseline_batch_{p_idx}")

    multihost_utils.sync_global_devices("baseline_done")

    # Save baseline results
    if is_master:
        passing = sum(1 for r in baseline_results if r["reward"] > 0)
        print(f"\n  Baseline complete: {passing}/{len(problems)} passing ({100*passing/len(problems):.1f}%)")
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline_results, f, indent=2, default=str)
        print(f"  Saved to {BASELINE_PATH}")

    # Clean up eSurge completely before Phase 2
    del esurge
    gc.collect()
    jax.clear_caches()
    gc.collect()
    multihost_utils.sync_global_devices("esurge_stopped")

    if is_master:
        print("  eSurge cleaned up, reloading model for Phase 2...")

    # Reload model fresh to avoid corrupted TPU state from eSurge
    del elm
    gc.collect()
    jax.clear_caches()

    elm = (
        ed.eLargeModel.from_pretrained(MODEL_ID)
        .set_dtype("bf16")
        .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
    )
    multihost_utils.sync_global_devices("model_reloaded")

else:
    if is_master:
        print("  Skipping eSurge build — going directly to Phase 2")


# ═════════════════════════════════════════════════════════════════════
# PHASE 2: PERTURBATION & REGRET COLLECTION
# ═════════════════════════════════════════════════════════════════════
if is_master:
    print(f"\n{'='*70}")
    print("PHASE 2: Forward-pass mode for perturbation & regret collection")
    print(f"{'='*70}")

# Monkey-patch EasyDeL output validation for multi-host eager mode
import easydel.infra.modeling_outputs as _mo
for _name, _cls in inspect.getmembers(_mo, inspect.isclass):
    if hasattr(_cls, '__post_init__'):
        _cls.__post_init__ = lambda self: None

if is_master:
    print("  Patched output validation for multi-host")

model_mesh = elm._model.config.mesh

# Filter to only problems with R* > 0 (skip already-failing baselines)
passing_indices = [i for i, r in enumerate(baseline_results) if r["reward"] > 0]

if is_master:
    print(f"  Processing {len(passing_indices)} passing prompts for perturbation")

# Regret data accumulators
regret_hidden_states = []   # List of (hidden_dim,) arrays
regret_layer_indices = []   # List of ints
regret_positions = []       # List of ints
regret_values = []          # List of floats


def directional_perturbation(logits, chosen_idx, k=PERTURB_TOP_K, strength=PERTURB_STRENGTH):
    """Apply directional perturbation: decrease chosen token, boost competitors.

    Args:
        logits: (vocab_size,) logit vector.
        chosen_idx: Index of the token chosen by greedy decode.
        k: Number of competitor tokens to boost.
        strength: Perturbation magnitude in units of logit std.

    Returns:
        Perturbed token index (sampled at low temperature from top-k).
    """
    logit_std = jnp.std(logits)

    # Decrease chosen token logit
    logits = logits.at[chosen_idx].add(-strength * logit_std)

    # Find top-k competitors (excluding chosen)
    # Set chosen to -inf temporarily for argsort
    masked = logits.at[chosen_idx].set(-jnp.inf)
    competitors = jnp.argsort(masked)[-k:][::-1]

    # Boost competitors proportionally
    boost = (strength / k) * logit_std
    for c_idx in range(k):
        logits = logits.at[competitors[c_idx]].add(boost)

    # Sample at low temperature from top-k (including original chosen)
    # Restrict to top-k tokens for sampling
    top_k_all = jnp.argsort(logits)[-(k + 1):]  # k competitors + possibly chosen
    top_k_logits = logits[top_k_all] / PERTURB_TEMP
    top_k_probs = jax.nn.softmax(top_k_logits)

    # Use deterministic key based on chosen_idx for reproducibility across workers
    rng = jax.random.PRNGKey(int(chosen_idx) * 1000 + k)
    sampled_local_idx = jax.random.categorical(rng, jnp.log(top_k_probs + 1e-10))
    return int(top_k_all[sampled_local_idx])


for pass_num, p_idx in enumerate(passing_indices):
    result = baseline_results[p_idx]
    problem = problems[p_idx]

    if is_master:
        print(f"\n{'─'*70}")
        print(f"[{pass_num+1}/{len(passing_indices)}] {problem.task_id} (R*={result['reward']})")

    # Tokenize full conversation (prompt + baseline response)
    conversation = build_code_prompt(problem)
    full_conversation = conversation + [{"role": "assistant", "content": result["generated_text"]}]
    input_ids = tokenizer.apply_chat_template(
        full_conversation, return_tensors="np", add_generation_prompt=False,
    )
    seq_len = input_ids.shape[1]

    # Find where the response starts
    prompt_only_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="np", add_generation_prompt=True,
    )
    response_start = prompt_only_ids.shape[1]
    response_len = seq_len - response_start

    if is_master:
        print(f"  Seq len: {seq_len}, response starts at: {response_start}, response tokens: {response_len}")

    # Full forward pass with hidden states
    if is_master:
        print(f"  Running baseline forward pass...")

    with model_mesh:
        baseline_out = elm._model(
            input_ids=jnp.array(input_ids),
            output_hidden_states=True,
        )

    baseline_logits = baseline_out.logits      # (1, seq_len, vocab)
    baseline_hidden = baseline_out.hidden_states  # tuple of (1, seq_len, hidden_dim)
    num_layers_total = len(baseline_hidden) - 1   # [0]=embeddings, [1..N]=layer outputs

    # Validate target layers
    valid_layers = [l for l in TARGET_LAYERS if l < num_layers_total]

    if is_master:
        print(f"  Model has {num_layers_total} layers, using: {valid_layers}")

    # Gather hidden states to numpy for target layers
    # (sharded JAX arrays can't be converted to numpy directly on multi-host)
    gathered_hidden = {}
    for l_idx in valid_layers:
        gathered_hidden[l_idx] = np.array(
            multihost_utils.process_allgather(baseline_hidden[l_idx + 1], tiled=True)
        )

    if is_master:
        print(f"  Gathered hidden states for {len(valid_layers)} layers to numpy")

    # Select response positions to perturb (subsample if too many)
    response_positions = list(range(response_start, seq_len - 1))  # exclude last token
    if len(response_positions) > MAX_POSITIONS_PER_PROMPT:
        # Evenly sample positions
        step = len(response_positions) / MAX_POSITIONS_PER_PROMPT
        response_positions = [response_positions[int(i * step)] for i in range(MAX_POSITIONS_PER_PROMPT)]

    if is_master:
        print(f"  Perturbing {len(response_positions)} positions × {len(valid_layers)} layers")

    # For each target layer × position: perturb, rollout, measure regret
    for layer_idx in valid_layers:
        # Get hidden state at this layer (already gathered to numpy)
        layer_hidden = gathered_hidden[layer_idx]  # (1, seq_len, hidden_dim)

        for pos in response_positions:
            # Get baseline logits at this position → greedy chosen token
            pos_logits = baseline_logits[0, pos, :]  # (vocab,)
            chosen_token = int(jnp.argmax(pos_logits))

            # Apply directional perturbation
            perturbed_token = directional_perturbation(pos_logits, chosen_token)

            # If perturbation didn't change the token, regret = 0 (skip expensive rollout)
            if perturbed_token == chosen_token:
                hidden_vec = layer_hidden[0, pos, :]
                regret_hidden_states.append(hidden_vec)
                regret_layer_indices.append(layer_idx)
                regret_positions.append(pos)
                regret_values.append(0.0)
                continue

            # Build perturbed sequence: replace token at pos, greedy rollout the rest
            perturbed_ids = np.array(input_ids[0, :pos + 1].tolist() + [perturbed_token])

            # Greedy rollout from pos+1 onward using forward passes
            current_ids = jnp.array(perturbed_ids).reshape(1, -1)

            with model_mesh:
                for rollout_step in range(seq_len - pos - 2):
                    # Cap rollout to avoid excessive compute
                    if current_ids.shape[1] >= seq_len:
                        break
                    rollout_out = elm._model(input_ids=current_ids)
                    next_logits = rollout_out.logits[0, -1, :]
                    next_token = int(jnp.argmax(next_logits))
                    current_ids = jnp.concatenate(
                        [current_ids, jnp.array([[next_token]])], axis=1
                    )
                    del rollout_out
                    if next_token == tokenizer.eos_token_id:
                        break

            # Decode perturbed response and execute tests
            perturbed_all_ids = np.array(current_ids[0])
            perturbed_response_ids = perturbed_all_ids[response_start:]
            perturbed_code_text = tokenizer.decode(perturbed_response_ids.tolist(), skip_special_tokens=True)
            perturbed_code = extract_code_from_response(perturbed_code_text)

            # Execute — master only
            perturbed_reward = 0.0
            if is_master:
                perturbed_reward, _ = execute_code(perturbed_code, problem.test_code, timeout=CODE_TIMEOUT)

            # Broadcast perturbed reward
            r_arr = jnp.array([perturbed_reward if is_master else 0.0])
            r_arr = multihost_utils.process_allgather(r_arr)
            perturbed_reward = float(r_arr.flatten()[0])

            # Compute regret
            regret = max(0.0, result["reward"] - perturbed_reward)

            # Store data point
            hidden_vec = np.array(layer_hidden[0, pos, :])
            regret_hidden_states.append(hidden_vec)
            regret_layer_indices.append(layer_idx)
            regret_positions.append(pos)
            regret_values.append(regret)

            del current_ids
            gc.collect()

        if is_master:
            nonzero = sum(1 for v in regret_values[-len(response_positions):] if v > 0)
            print(f"    Layer {layer_idx}: {nonzero}/{len(response_positions)} positions with regret > 0")

    # Cleanup after each prompt
    del baseline_out, baseline_logits, baseline_hidden, gathered_hidden
    gc.collect()
    multihost_utils.sync_global_devices(f"perturb_done_{pass_num}")

    if is_master:
        print(f"  Total regret samples so far: {len(regret_values)}")

multihost_utils.sync_global_devices("perturbation_complete")

# Save regret dataset
if is_master:
    print(f"\n{'='*70}")
    print(f"Saving regret dataset: {len(regret_values)} samples")

    regret_data = {
        "hidden_states": np.stack(regret_hidden_states),  # (N, hidden_dim)
        "layer_indices": np.array(regret_layer_indices),   # (N,)
        "positions": np.array(regret_positions),           # (N,)
        "regret_values": np.array(regret_values),          # (N,)
    }
    np.savez(REGRET_DATA_PATH, **regret_data)
    print(f"  Saved to {REGRET_DATA_PATH}")

    nonzero_count = np.sum(np.array(regret_values) > 0)
    print(f"  Non-zero regret: {nonzero_count}/{len(regret_values)} ({100*nonzero_count/max(len(regret_values),1):.1f}%)")
    print(f"  Mean regret: {np.mean(regret_values):.4f}")
    print(f"  Max regret: {np.max(regret_values):.4f}")


# ═════════════════════════════════════════════════════════════════════
# PHASE 3: TRAIN REGRET ESTIMATOR
# ═════════════════════════════════════════════════════════════════════
if is_master:
    print(f"\n{'='*70}")
    print("PHASE 3: Training regret estimator MLP")
    print(f"{'='*70}")

# Free model memory — we only need CPU/single-device for MLP training
del elm
gc.collect()
multihost_utils.sync_global_devices("model_freed")

# Only master trains the estimator (it's a small MLP, no need for multi-host)
if is_master:
    import optax
    from regret_estimator import create_estimator, save_estimator
    from flax import nnx

    # Load regret dataset
    data = np.load(REGRET_DATA_PATH)
    hidden_states = jnp.array(data["hidden_states"])
    layer_indices = jnp.array(data["layer_indices"])
    positions = jnp.array(data["positions"])
    regret_targets = jnp.array(data["regret_values"])

    N = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    print(f"  Dataset: {N} samples, hidden_dim={hidden_dim}")

    # Normalize inputs
    num_layers = max(int(jnp.max(layer_indices)) + 1, 1)
    max_pos = max(int(jnp.max(positions)) + 1, 1)
    layer_norm = layer_indices.astype(jnp.float32).reshape(-1, 1) / (num_layers - 1)
    pos_norm = positions.astype(jnp.float32).reshape(-1, 1) / (max_pos - 1)

    # Train/val split
    perm = np.random.RandomState(42).permutation(N)
    split = int(N * TRAIN_SPLIT)
    train_idx, val_idx = perm[:split], perm[split:]

    # Create model and optimizer
    model = create_estimator(hidden_dim, seed=42)
    optimizer = nnx.Optimizer(model, optax.adam(ESTIMATOR_LR))

    def loss_fn(model, h, l, p, targets):
        preds = model(h, l, p)
        return jnp.mean((preds.squeeze(-1) - targets) ** 2)

    @nnx.jit
    def train_step(model, optimizer, h, l, p, targets):
        loss, grads = nnx.value_and_grad(loss_fn)(model, h, l, p, targets)
        optimizer.update(grads)
        return loss

    print(f"  Training for {ESTIMATOR_EPOCHS} epochs, batch_size={ESTIMATOR_BATCH}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    best_val_loss = float("inf")

    for epoch in range(ESTIMATOR_EPOCHS):
        # Shuffle training data
        epoch_perm = np.random.RandomState(epoch).permutation(len(train_idx))
        shuffled = train_idx[epoch_perm]

        # Training loop
        epoch_losses = []
        for batch_start in range(0, len(shuffled), ESTIMATOR_BATCH):
            batch_end = min(batch_start + ESTIMATOR_BATCH, len(shuffled))
            idx = shuffled[batch_start:batch_end]

            loss = train_step(
                model, optimizer,
                hidden_states[idx], layer_norm[idx], pos_norm[idx],
                regret_targets[idx],
            )
            epoch_losses.append(float(loss))

        train_loss = np.mean(epoch_losses)

        # Validation
        val_preds = model(hidden_states[val_idx], layer_norm[val_idx], pos_norm[val_idx])
        val_loss = float(jnp.mean((val_preds.squeeze(-1) - regret_targets[val_idx]) ** 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_estimator(model, ESTIMATOR_PATH)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                  f"{' *best*' if val_loss <= best_val_loss else ''}")

    print(f"\n  Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"  Model saved to {ESTIMATOR_PATH}.npz")

multihost_utils.sync_global_devices("training_complete")

# ═════════════════════════════════════════════════════════════════════
# CLEANUP
# ═════════════════════════════════════════════════════════════════════
if is_master:
    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"  Baseline results: {BASELINE_PATH}")
    print(f"  Regret dataset:   {REGRET_DATA_PATH}")
    print(f"  Estimator model:  {ESTIMATOR_PATH}.npz")
    print(f"{'='*70}")

gc.collect()
multihost_utils.sync_global_devices("ready_to_kill")
jax.distributed.shutdown()
sys.exit(0)
