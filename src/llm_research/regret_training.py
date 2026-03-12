"""
Regret-Aware Early Exit Training Pipeline
===========================================
Three-phase pipeline:
  Phase 1: Greedy baseline generation on HumanEval/MBPP → reward R*
  Phase 2a: Forward passes → hidden states + perturbation planning
  Phase 2b: eSurge greedy rollouts → code testing → regret collection
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
TARGET_LAYERS = [8, 24, 40]               # Sample 3 layers (early/mid/late)
PERTURB_STRENGTH = 2.0       # Logit perturbation magnitude
PERTURB_TEMP = 0.3           # Temperature for perturbed sampling
PERTURB_TOP_K = 3            # Top-k competitors for perturbation
MAX_POSITIONS_PER_PROMPT = 20  # Token positions to perturb per prompt
ROLLOUT_MAX_TOKENS = 256     # Max tokens for perturbed rollout generation
BATCH_SIZE = 4               # Prompts to process before GC
CODE_TIMEOUT = 10            # Seconds for code execution
REGRET_FOCAL_ALPHA = 10.0    # Focal loss weight for non-zero regret samples

# Regret estimator training
ESTIMATOR_LR = 1e-4
ESTIMATOR_EPOCHS = 50
ESTIMATOR_BATCH = 256
TRAIN_SPLIT = 0.8

# Output paths
BASELINE_PATH = "regret_baseline_results.json"
PHASE2A_HIDDEN_PATH = "phase2a_hidden_states.npz"
PHASE2A_PERTURB_PATH = "phase2a_perturbations.json"
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
    print(f"  Positions/prompt: {MAX_POSITIONS_PER_PROMPT}")
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

# Always configure eSurge to trigger model weight conversion into JAX
# (elm._model is None until set_esurge initializes it)
elm = elm.set_esurge(max_model_len=4096, max_num_seqs=4)


# ═════════════════════════════════════════════════════════════════════
# PHASE 1: BASELINE GENERATION
# ═════════════════════════════════════════════════════════════════════

# Check if Phase 1 is already complete (checkpoint with all problems)
# ALL workers load checkpoint — needed so all workers have baseline_results for Phase 2
phase1_complete = False
baseline_results = []
if os.path.exists(BASELINE_PATH):
    try:
        with open(BASELINE_PATH) as f:
            baseline_results = json.load(f)
        if len(baseline_results) >= len(problems):
            phase1_complete = True
            if is_master:
                print(f"\n  Phase 1 already complete: {len(baseline_results)} results loaded from checkpoint")
        else:
            if is_master:
                print(f"\n  Partial checkpoint: {len(baseline_results)}/{len(problems)}")
    except Exception as e:
        if is_master:
            print(f"  Could not load checkpoint: {e}, starting fresh")
        baseline_results = []

# Master checks if Phase 1 is done and broadcasts to all workers
# (master = process_index 0, which is what broadcast_one_to_all sends from)
p1_flag = jnp.array([1.0 if (is_master and phase1_complete) else 0.0])
p1_flag = multihost_utils.broadcast_one_to_all(p1_flag)
phase1_complete = bool(p1_flag[0] > 0.5)

# If Phase 1 is complete, broadcast baseline_results from master to all workers
# (only master may have the checkpoint file from a previous run)
if phase1_complete:
    # Broadcast data length first
    if is_master:
        json_bytes = json.dumps(baseline_results, default=str).encode("utf-8")
        data_len = len(json_bytes)
    else:
        data_len = 0
    len_arr = multihost_utils.broadcast_one_to_all(jnp.array([data_len]))
    data_len = int(len_arr[0])

    # Broadcast the JSON data as uint8 array
    if is_master:
        data_arr = jnp.array(np.frombuffer(json_bytes, dtype=np.uint8))
    else:
        data_arr = jnp.zeros(data_len, dtype=jnp.uint8)
    data_arr = multihost_utils.broadcast_one_to_all(data_arr)

    if not is_master:
        raw = np.array(data_arr, dtype=np.uint8).tobytes()
        baseline_results = json.loads(raw.decode("utf-8"))
        # Save locally for future runs
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline_results, f, indent=2, default=str)

    if is_master:
        print(f"  Broadcast {data_len} bytes of baseline_results to all workers")

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

        # Periodic GC and checkpoint saving (all workers save their own copy)
        if (p_idx + 1) % BATCH_SIZE == 0:
            gc.collect()
            if (p_idx + 1) % (BATCH_SIZE * 5) == 0:
                with open(BASELINE_PATH, "w") as f:
                    json.dump(baseline_results, f, indent=2, default=str)
                if is_master:
                    print(f"  [Checkpoint saved: {len(baseline_results)}/{len(problems)}]")
            multihost_utils.sync_global_devices(f"baseline_batch_{p_idx}")

    multihost_utils.sync_global_devices("baseline_done")

    # Save baseline results (all workers save their own copy for Phase 2 resume)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)
    if is_master:
        passing = sum(1 for r in baseline_results if r["reward"] > 0)
        print(f"\n  Baseline complete: {passing}/{len(problems)} passing ({100*passing/len(problems):.1f}%)")
        print(f"  Saved to {BASELINE_PATH}")

    # Clean up eSurge completely before Phase 2a
    del esurge
    gc.collect()
    jax.clear_caches()
    gc.collect()
    multihost_utils.sync_global_devices("esurge_stopped")

    if is_master:
        print("  eSurge cleaned up, reloading model for Phase 2a...")

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
        print("  Phase 1 complete — building model for Phase 2a only")

    # Build eSurge just to initialize _model, then immediately tear it down
    # (no inference run = no TPU state corruption)
    esurge = elm.build_esurge()
    del esurge
    gc.collect()
    jax.clear_caches()
    gc.collect()
    multihost_utils.sync_global_devices("model_ready_phase2")

    if is_master:
        print("  Model initialized and eSurge cleaned up (no inference was run)")


# ═════════════════════════════════════════════════════════════════════
# PHASE 2a: FORWARD PASSES — HIDDEN STATES & PERTURBATION PLANNING
# ═════════════════════════════════════════════════════════════════════

# Check completion status of each sub-phase
phase2_complete = is_master and os.path.exists(REGRET_DATA_PATH)
p2_flag = jnp.array([1.0 if phase2_complete else 0.0])
p2_flag = multihost_utils.broadcast_one_to_all(p2_flag)
phase2_complete = bool(p2_flag[0] > 0.5)

phase2a_complete = is_master and os.path.exists(PHASE2A_HIDDEN_PATH) and os.path.exists(PHASE2A_PERTURB_PATH)
p2a_flag = jnp.array([1.0 if phase2a_complete else 0.0])
p2a_flag = multihost_utils.broadcast_one_to_all(p2a_flag)
phase2a_complete = bool(p2a_flag[0] > 0.5)

if phase2_complete:
    if is_master:
        print(f"\n  Phase 2 already complete: {REGRET_DATA_PATH} exists, skipping to Phase 3")

if not phase2_complete and not phase2a_complete:
    if is_master:
        print(f"\n{'='*70}")
        print("PHASE 2a: Forward passes — hidden states & perturbation planning")
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
        print(f"  Processing {len(passing_indices)} passing prompts")

    # Accumulators for Phase 2a output
    all_hidden_states = []    # (hidden_dim,) arrays — one per layer × position
    all_layer_indices = []
    all_positions = []
    all_perturbations = []    # List of dicts: {p_idx, pos, perturbed_token, prefix_text, ...}

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

        if is_master:
            print(f"  Seq len: {seq_len}, response starts: {response_start}, response tokens: {seq_len - response_start}")

        # Full forward pass with hidden states
        if is_master:
            print(f"  Running forward pass...")

        with model_mesh:
            baseline_out = elm._model(
                input_ids=jnp.array(input_ids),
                output_hidden_states=True,
            )

        baseline_logits = baseline_out.logits
        baseline_hidden = baseline_out.hidden_states
        num_layers_total = len(baseline_hidden) - 1

        valid_layers = [l for l in TARGET_LAYERS if l < num_layers_total]

        # Gather hidden states and logits to numpy
        gathered_hidden = {}
        for l_idx in valid_layers:
            gathered_hidden[l_idx] = np.array(
                multihost_utils.process_allgather(baseline_hidden[l_idx + 1], tiled=True)
            )

        baseline_logits_np = np.array(
            multihost_utils.process_allgather(baseline_logits, tiled=True)
        )

        # Select response positions to perturb
        response_positions = list(range(response_start, seq_len - 1))
        if len(response_positions) > MAX_POSITIONS_PER_PROMPT:
            step = len(response_positions) / MAX_POSITIONS_PER_PROMPT
            response_positions = [response_positions[int(i * step)] for i in range(MAX_POSITIONS_PER_PROMPT)]

        if is_master:
            print(f"  {len(response_positions)} positions × {len(valid_layers)} layers")

        # Compute perturbations and collect hidden states
        for pos in response_positions:
            pos_logits = baseline_logits_np[0, pos, :]
            chosen_token = int(np.argmax(pos_logits))

            # Deterministic perturbation (pure numpy)
            logit_std = float(np.std(pos_logits))
            perturbed_logits = pos_logits.copy()
            perturbed_logits[chosen_token] -= PERTURB_STRENGTH * logit_std
            masked = perturbed_logits.copy()
            masked[chosen_token] = -np.inf
            competitors = np.argsort(masked)[-PERTURB_TOP_K:][::-1]
            boost = (PERTURB_STRENGTH / PERTURB_TOP_K) * logit_std
            for c in competitors:
                perturbed_logits[c] += boost
            top_k_all = np.argsort(perturbed_logits)[-(PERTURB_TOP_K + 1):]
            top_k_logits = perturbed_logits[top_k_all] / PERTURB_TEMP
            top_k_logits -= top_k_logits.max()
            top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
            rng = np.random.RandomState(int(chosen_token) * 1000 + PERTURB_TOP_K)
            perturbed_token = int(top_k_all[rng.choice(len(top_k_probs), p=top_k_probs)])

            same_token = (perturbed_token == chosen_token)

            # Collect hidden states for each target layer at this position
            for layer_idx in valid_layers:
                layer_hidden = gathered_hidden[layer_idx]
                all_hidden_states.append(np.float32(layer_hidden[0, pos, :]))
                all_layer_indices.append(layer_idx)
                all_positions.append(pos)

            # Build the perturbed prefix text for Phase 2b rollout
            # (one rollout per position, shared across layers)
            if not same_token:
                perturbed_prefix_ids = np.array(input_ids[0][:pos + 1])
                perturbed_prefix_ids[pos] = perturbed_token
                prefix_response_text = tokenizer.decode(
                    perturbed_prefix_ids[response_start:].tolist(),
                    skip_special_tokens=True,
                )
            else:
                prefix_response_text = ""

            all_perturbations.append({
                "p_idx": int(p_idx),
                "pos": int(pos),
                "response_start": int(response_start),
                "chosen_token": int(chosen_token),
                "perturbed_token": int(perturbed_token),
                "same_token": same_token,
                "prefix_response_text": prefix_response_text,
                "baseline_reward": float(result["reward"]),
                "num_layers": len(valid_layers),
            })

        # Cleanup per prompt
        del baseline_out, baseline_logits, baseline_logits_np, baseline_hidden, gathered_hidden
        gc.collect()
        multihost_utils.sync_global_devices(f"phase2a_{pass_num}")

        if is_master and (pass_num + 1) % 10 == 0:
            print(f"  Progress: {pass_num+1}/{len(passing_indices)}, samples: {len(all_perturbations)}")

    multihost_utils.sync_global_devices("phase2a_complete")

    # Save Phase 2a data (master only — hidden states are already gathered)
    if is_master:
        print(f"\n  Saving Phase 2a data: {len(all_perturbations)} perturbations, {len(all_hidden_states)} hidden states")

        np.savez(
            PHASE2A_HIDDEN_PATH,
            hidden_states=np.stack(all_hidden_states),
            layer_indices=np.array(all_layer_indices),
            positions=np.array(all_positions),
        )
        with open(PHASE2A_PERTURB_PATH, "w") as f:
            json.dump(all_perturbations, f, indent=2)

        n_need_rollout = sum(1 for p in all_perturbations if not p["same_token"])
        print(f"  Saved to {PHASE2A_HIDDEN_PATH} and {PHASE2A_PERTURB_PATH}")
        print(f"  Perturbations needing rollout: {n_need_rollout}/{len(all_perturbations)}")

    phase2a_complete = True

if is_master:
    print(f"\n  Phase 2a {'complete' if phase2a_complete else 'skipped (Phase 2 already done)'}")


# ═════════════════════════════════════════════════════════════════════
# PHASE 2b: eSURGE ROLLOUTS — GREEDY CONTINUATION & REGRET
# ═════════════════════════════════════════════════════════════════════

if not phase2_complete and phase2a_complete:
    if is_master:
        print(f"\n{'='*70}")
        print("PHASE 2b: eSurge greedy rollouts for perturbed sequences")
        print(f"{'='*70}")

    # Reload model fresh for eSurge (forward passes with output_hidden_states
    # modify the model graph, preventing eSurge rebuild)
    if is_master:
        print("  Reloading model for eSurge...")

    del elm
    gc.collect()
    jax.clear_caches()
    gc.collect()
    multihost_utils.sync_global_devices("model_freed_for_2b")

    elm = (
        ed.eLargeModel.from_pretrained(MODEL_ID)
        .set_dtype("bf16")
        .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
        .set_esurge(max_model_len=4096, max_num_seqs=4)
    )

    if is_master:
        print("  Building eSurge for rollouts...")

    esurge = elm.build_esurge()

    if is_master:
        print("  eSurge ready")

    # Load Phase 2a perturbation data (master only, then broadcast)
    if is_master:
        with open(PHASE2A_PERTURB_PATH) as f:
            perturbations = json.load(f)
        perturb_json = json.dumps(perturbations).encode("utf-8")
        perturb_len = len(perturb_json)
    else:
        perturb_len = 0

    len_arr = multihost_utils.broadcast_one_to_all(jnp.array([perturb_len]))
    perturb_len = int(len_arr[0])

    if is_master:
        data_arr = jnp.array(np.frombuffer(perturb_json, dtype=np.uint8))
    else:
        data_arr = jnp.zeros(perturb_len, dtype=jnp.uint8)
    data_arr = multihost_utils.broadcast_one_to_all(data_arr)

    if not is_master:
        perturbations = json.loads(np.array(data_arr, dtype=np.uint8).tobytes().decode("utf-8"))

    if is_master:
        print(f"  Loaded {len(perturbations)} perturbations")

    # Process rollouts — one per perturbation position (not per layer)
    # Regret is shared across layers at the same position
    regret_values = []  # One per perturbation (not per layer)
    t_rollout_start = time.time()

    for pert_idx, pert in enumerate(perturbations):
        if pert["same_token"]:
            regret_values.append(0.0)
            continue

        # Construct prompt for continuation
        problem = problems[pert["p_idx"]]
        conversation = build_code_prompt(problem)
        prefix_text = pert["prefix_response_text"]

        # Embed the perturbed prefix in the prompt for continuation
        continuation_conv = [{"role": "user", "content": (
            conversation[0]["content"]
            + "\n\nIMPORTANT: Your solution MUST begin exactly with this code and continue from where it ends. "
            + "Do NOT restart or rewrite the beginning:\n```python\n"
            + prefix_text
        )}]

        # Generate continuation via eSurge
        cont_text = ""
        for output in esurge.chat(
            continuation_conv,
            sampling_params=ed.SamplingParams(
                max_tokens=ROLLOUT_MAX_TOKENS,
                temperature=0.0,
            ),
            stream=True,
        ):
            cont_text += output.delta_text

        # Extract code and test
        perturbed_reward = 0.0
        if is_master:
            perturbed_code = extract_code_from_response(cont_text)
            if not perturbed_code.strip():
                # Model may have included the prefix — try combining
                perturbed_code = extract_code_from_response(prefix_text + cont_text)
            perturbed_reward, _ = execute_code(perturbed_code, problem.test_code, timeout=CODE_TIMEOUT)

        regret = max(0.0, pert["baseline_reward"] - perturbed_reward)
        regret_values.append(regret)

        if is_master and (pert_idx + 1) % 50 == 0:
            elapsed = time.time() - t_rollout_start
            nonzero = sum(1 for v in regret_values if v > 0)
            print(f"  [{pert_idx+1}/{len(perturbations)}] {elapsed:.0f}s, "
                  f"non-zero regret: {nonzero}/{len(regret_values)} ({100*nonzero/max(len(regret_values),1):.1f}%)")

    # Broadcast regret values from master
    r_arr = jnp.array(regret_values if is_master else [0.0] * len(regret_values))
    r_arr = multihost_utils.broadcast_one_to_all(r_arr)
    regret_values = r_arr.tolist()

    multihost_utils.sync_global_devices("phase2b_complete")

    # Clean up eSurge
    del esurge
    gc.collect()

    # Assemble final regret dataset: expand regret per position → per layer × position
    if is_master:
        elapsed_total = time.time() - t_rollout_start
        print(f"\n  Rollouts complete in {elapsed_total:.0f}s")

        # Load Phase 2a hidden states
        h_data = np.load(PHASE2A_HIDDEN_PATH)
        hidden_states_all = h_data["hidden_states"]
        layer_indices_all = h_data["layer_indices"]
        positions_all = h_data["positions"]

        # Each perturbation maps to num_layers hidden state entries
        # perturbations[i] → hidden states at indices [i*num_layers : (i+1)*num_layers]
        num_layers = perturbations[0]["num_layers"]
        expanded_regret = []
        for pert_idx, pert in enumerate(perturbations):
            regret = regret_values[pert_idx]
            for _ in range(num_layers):
                expanded_regret.append(regret)

        expanded_regret = np.array(expanded_regret)

        assert len(expanded_regret) == len(hidden_states_all), \
            f"Mismatch: {len(expanded_regret)} regret vs {len(hidden_states_all)} hidden states"

        # Save final regret dataset
        print(f"\n{'='*70}")
        print(f"Saving regret dataset: {len(expanded_regret)} samples")

        np.savez(
            REGRET_DATA_PATH,
            hidden_states=hidden_states_all,
            layer_indices=layer_indices_all,
            positions=positions_all,
            regret_values=expanded_regret,
        )
        print(f"  Saved to {REGRET_DATA_PATH}")

        nonzero_count = np.sum(expanded_regret > 0)
        print(f"  Non-zero regret: {nonzero_count}/{len(expanded_regret)} ({100*nonzero_count/len(expanded_regret):.1f}%)")
        print(f"  Mean regret: {np.mean(expanded_regret):.4f}")
        print(f"  Max regret: {np.max(expanded_regret):.4f}")


# ═════════════════════════════════════════════════════════════════════
# PHASE 3: TRAIN REGRET ESTIMATOR
# ═════════════════════════════════════════════════════════════════════
if is_master:
    print(f"\n{'='*70}")
    print("PHASE 3: Training regret estimator MLP")
    print(f"{'='*70}")

# Free model memory — we only need CPU/single-device for MLP training
try:
    del elm
except NameError:
    pass
gc.collect()
multihost_utils.sync_global_devices("model_freed")

# Only master trains the estimator (it's a small MLP, no need for multi-host)
if is_master:
    import optax
    from regret_estimator import create_estimator, save_estimator
    from flax import nnx

    # Load regret dataset
    data = np.load(REGRET_DATA_PATH, allow_pickle=False)
    hs = data["hidden_states"]
    # Handle bfloat16 saved as raw bytes (dtype |V2) from earlier runs
    if hs.dtype.kind not in ('f', 'i', 'u'):
        hs_uint16 = hs.view(np.uint16)
        hs = np.array(jnp.array(hs_uint16).view(jnp.bfloat16).astype(jnp.float32))
    hidden_states = jnp.array(hs, dtype=jnp.float32)
    layer_indices = jnp.array(data["layer_indices"])
    positions = jnp.array(data["positions"])
    regret_targets = jnp.array(data["regret_values"])

    N = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    print(f"  Dataset: {N} samples, hidden_dim={hidden_dim}")

    nonzero_frac = float(jnp.mean(regret_targets > 0))
    print(f"  Non-zero regret fraction: {nonzero_frac:.3f}")

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
    optimizer = nnx.Optimizer(model, optax.adam(ESTIMATOR_LR), wrt=nnx.Param)

    def loss_fn(model, h, l, p, targets):
        preds = model(h, l, p).squeeze(-1)
        residuals = (preds - targets) ** 2
        # Focal weighting: upweight non-zero regret samples
        weights = jnp.where(targets > 0, REGRET_FOCAL_ALPHA, 1.0)
        return jnp.mean(weights * residuals)

    def train_step(model, optimizer, h, l, p, targets):
        loss, grads = nnx.value_and_grad(loss_fn)(model, h, l, p, targets)
        optimizer.update(model, grads)
        return loss

    print(f"  Training for {ESTIMATOR_EPOCHS} epochs, batch_size={ESTIMATOR_BATCH}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"  Focal alpha: {REGRET_FOCAL_ALPHA} (upweighting non-zero regret)")

    best_val_loss = float("inf")

    for epoch in range(ESTIMATOR_EPOCHS):
        epoch_perm = np.random.RandomState(epoch).permutation(len(train_idx))
        shuffled = train_idx[epoch_perm]

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

        # Validation (use unweighted MSE for val to measure real prediction quality)
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
