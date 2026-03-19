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

import sys
# Force unbuffered output so tee gets lines immediately
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np
import gc
import json
import inspect
import time
import os

# ── CONFIGURABLE PARAMETERS ──────────────────────────────────────────
# All shared constants are in config.py; import them here.
from config import (  # noqa: E402
    MODEL_ID, MAX_GEN_TOKENS, TARGET_LAYERS,
    PERTURB_STRENGTH, PERTURB_TEMP, PERTURB_TOP_K,
    MAX_POSITIONS_PER_PROMPT, ROLLOUT_MAX_TOKENS, CODE_TIMEOUT,
    ESTIMATOR_LR, ESTIMATOR_EPOCHS, ESTIMATOR_BATCH,
    ESTIMATOR_PATIENCE, ESTIMATOR_DROPOUT, ESTIMATOR_WEIGHT_DECAY,
    LABEL_SMOOTHING, TRAIN_SPLIT,
    BASELINE_PATH, PHASE2A_HIDDEN_PATH, PHASE2A_PERTURB_PATH,
    REGRET_DATA_PATH, PHASE2A_FAILING_PATH, ESTIMATOR_PATH,
    PHASE2B_CHECKPOINT_PATH, PHASE2B_CHECKPOINT_INTERVAL, BATCH_SIZE,
    V2_MODE, V2_SKIP_PHASE2B, V2_NUM_EXTRA_FEATURES, SYNTAX_TOKENS,
)

# Training-specific (not shared)
REGRET_FOCAL_ALPHA = 10.0    # Focal loss weight for non-zero regret samples

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

# Pre-check completion status BEFORE model load to pick the right config.
# This avoids loading the model twice (which OOMs due to HBM fragmentation).
_p2_done = is_master and os.path.exists(REGRET_DATA_PATH)
_p2a_done = is_master and os.path.exists(PHASE2A_HIDDEN_PATH) and os.path.exists(PHASE2A_PERTURB_PATH)
_p2_flag = multihost_utils.broadcast_one_to_all(jnp.array([1.0 if _p2_done else 0.0]))
_p2a_flag = multihost_utils.broadcast_one_to_all(jnp.array([1.0 if _p2a_done else 0.0]))
_need_phase2b_only = bool(_p2a_flag[0] > 0.5) and not bool(_p2_flag[0] > 0.5)

if _need_phase2b_only:
    # Phase 2a done, Phase 2b needed — load model directly with Phase 2b eSurge config
    # (skip the full-size config to avoid double-loading OOM)
    if is_master:
        print("  Phase 2a complete, loading model directly for Phase 2b rollouts...")
    elm = (
        ed.eLargeModel.from_pretrained(MODEL_ID)
        .set_dtype("bf16")
        .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
        .set_esurge(max_model_len=512, max_num_seqs=1, hbm_utilization=0.5)
    )
else:
    elm = (
        ed.eLargeModel.from_pretrained(MODEL_ID)
        .set_dtype("bf16")
        .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
    )
    # Configure eSurge to trigger model weight conversion into JAX
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
            reward, exec_info = execute_code(code, problem.test_code, timeout=CODE_TIMEOUT, partial=True)
            print(f" {num_tokens} tok, {elapsed:.1f}s, R*={reward:.2f}")
            if reward < 1.0 and exec_info.get("stderr"):
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

    passing_indices = [i for i, r in enumerate(baseline_results) if r["reward"] > 0]
    failing_indices = [i for i, r in enumerate(baseline_results) if r["reward"] == 0]

    if is_master:
        print(f"  Passing prompts: {len(passing_indices)}, Failing: {len(failing_indices)}")

    # ── Helper: run forward pass and collect hidden states + logit features ──
    def _forward_pass_collect(p_idx, result, problem, pass_label):
        """Run forward pass, return (hidden_states, layer_indices, positions, entropies, margins)
        for sampled response positions across target layers."""
        conversation = build_code_prompt(problem)
        full_conversation = conversation + [{"role": "assistant", "content": result["generated_text"]}]
        input_ids = tokenizer.apply_chat_template(
            full_conversation, return_tensors="np", add_generation_prompt=False,
        )
        seq_len = input_ids.shape[1]

        prompt_only_ids = tokenizer.apply_chat_template(
            conversation, return_tensors="np", add_generation_prompt=True,
        )
        response_start = prompt_only_ids.shape[1]

        if is_master:
            print(f"  Seq len: {seq_len}, response: {seq_len - response_start} tokens")

        with model_mesh:
            out = elm._model(
                input_ids=jnp.array(input_ids),
                output_hidden_states=True,
            )

        logits = out.logits
        hidden = out.hidden_states
        num_layers_total = len(hidden) - 1
        valid_layers = [l for l in TARGET_LAYERS if l < num_layers_total]

        gathered_hidden = {}
        for l_idx in valid_layers:
            gathered_hidden[l_idx] = np.array(
                multihost_utils.process_allgather(hidden[l_idx + 1], tiled=True)
            )
        logits_np = np.array(
            multihost_utils.process_allgather(logits, tiled=True)
        )

        # Select response positions
        response_positions = list(range(response_start, seq_len - 1))
        if len(response_positions) > MAX_POSITIONS_PER_PROMPT:
            step = len(response_positions) / MAX_POSITIONS_PER_PROMPT
            response_positions = [response_positions[int(i * step)] for i in range(MAX_POSITIONS_PER_PROMPT)]

        hs_list, li_list, pos_list, ent_list, mar_list = [], [], [], [], []

        for pos in response_positions:
            pos_logits = logits_np[0, pos, :]

            # Compute logit features: entropy and margin
            shifted = pos_logits - pos_logits.max()
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            sorted_logits = np.sort(pos_logits)
            margin = float(sorted_logits[-1] - sorted_logits[-2])

            for layer_idx in valid_layers:
                hs_list.append(np.float32(gathered_hidden[layer_idx][0, pos, :]))
                li_list.append(layer_idx)
                pos_list.append(pos)
                ent_list.append(entropy)
                mar_list.append(margin)

        # Cleanup
        del out, logits, logits_np, hidden, gathered_hidden
        gc.collect()

        return (hs_list, li_list, pos_list, ent_list, mar_list,
                input_ids, response_start, response_positions, valid_layers, logits_np if False else None)

    # ── V2 helpers: logit probing + convergence + token type ──
    def _rms_norm_np(x, weight, eps=1e-6):
        """Apply RMSNorm in numpy: x * weight / sqrt(mean(x^2) + eps)."""
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return (x / rms) * weight

    def _compute_v2_features(gathered_hidden, final_logits_pos, layer_idx, prev_layer_idx,
                             valid_layers, pos, lm_head_kernel, final_norm_weight,
                             token_str):
        """Compute V2 features for a single (position, layer) pair.

        Returns dict with: agreement, confidence, kl_div, int_entropy,
                          cos_sim, rel_change, proj_final, is_syntax, is_whitespace
        """
        h = gathered_hidden[layer_idx][0, pos, :].astype(np.float32)

        # ── Logit probing: intermediate logits via lm_head ──
        h_normed = _rms_norm_np(h, final_norm_weight)
        intermediate_logits = h_normed @ lm_head_kernel  # (vocab_size,)

        # Agreement: argmax match with final layer
        agreement = float(np.argmax(intermediate_logits) == np.argmax(final_logits_pos))

        # Confidence: softmax probability of top intermediate token
        int_shifted = intermediate_logits - intermediate_logits.max()
        int_probs = np.exp(int_shifted) / np.sum(np.exp(int_shifted))
        confidence = float(int_probs[np.argmax(intermediate_logits)])

        # KL divergence: KL(final || intermediate)
        fin_shifted = final_logits_pos - final_logits_pos.max()
        fin_probs = np.exp(fin_shifted) / np.sum(np.exp(fin_shifted))
        kl_raw = float(np.sum(fin_probs * np.log((fin_probs + 1e-10) / (int_probs + 1e-10))))
        kl_div = np.log1p(max(kl_raw, 0.0))  # log-scale to prevent outliers

        # Intermediate entropy
        int_entropy = float(-np.sum(int_probs * np.log(int_probs + 1e-10)))

        # ── Convergence metrics ──
        if prev_layer_idx is not None and prev_layer_idx in gathered_hidden:
            h_prev = gathered_hidden[prev_layer_idx][0, pos, :].astype(np.float32)
            h_norm = np.linalg.norm(h)
            h_prev_norm = np.linalg.norm(h_prev)

            # Cosine similarity (log-scaled to avoid saturation near 1.0)
            cos_raw = np.dot(h, h_prev) / (h_norm * h_prev_norm + 1e-10)
            cos_sim = -np.log1p(-np.clip(cos_raw, -1.0, 1.0 - 1e-7))  # log(1/(1-cos))

            # Relative L2 change (log-scaled)
            rel_raw = np.linalg.norm(h - h_prev) / (h_prev_norm + 1e-10)
            rel_change = np.log1p(rel_raw)
        else:
            cos_sim = 0.0
            rel_change = 1.0  # max change for first layer

        # Projection onto final hidden state direction
        final_layer = max(valid_layers)
        h_final = gathered_hidden[final_layer][0, pos, :].astype(np.float32)
        h_final_norm = np.linalg.norm(h_final) + 1e-10
        proj_final = float(np.dot(h, h_final) / (np.linalg.norm(h) * h_final_norm + 1e-10))

        # ── Token type features ──
        tok_stripped = token_str.strip()
        is_syntax = float(tok_stripped in SYNTAX_TOKENS)
        is_whitespace = float(tok_stripped == '')

        return {
            "agreement": agreement,
            "confidence": confidence,
            "kl_div": float(kl_div),
            "int_entropy": int_entropy,
            "cos_sim": float(cos_sim),
            "rel_change": float(rel_change),
            "proj_final": proj_final,
            "is_syntax": is_syntax,
            "is_whitespace": is_whitespace,
        }

    # ── Extract lm_head + final norm params for V2 logit probing ──
    lm_head_kernel = None
    final_norm_weight = None
    if V2_MODE:
        if is_master:
            print("  V2 MODE: extracting lm_head and final norm parameters...")

        with model_mesh:
            # lm_head.kernel: (hidden_dim, vocab_size)
            _lm_k = elm._model.lm_head.kernel
            lm_head_kernel_jax = multihost_utils.process_allgather(_lm_k, tiled=True)
            # Cast bf16 → f32 in JAX before converting to numpy (numpy can't handle bf16)
            lm_head_kernel = np.array(lm_head_kernel_jax.astype(jnp.float32))
            del lm_head_kernel_jax

            # Final RMSNorm weight: (hidden_dim,)
            _norm_w = elm._model.model.norm.kernel
            norm_w_jax = multihost_utils.process_allgather(_norm_w, tiled=True)
            final_norm_weight = np.array(norm_w_jax.astype(jnp.float32))
            del norm_w_jax

        if is_master:
            print(f"    lm_head kernel: {lm_head_kernel.shape}")
            print(f"    final norm weight: {final_norm_weight.shape}")

    # ── Process passing baselines (perturbation planning + hidden states) ──
    all_hidden_states = []
    all_layer_indices = []
    all_positions = []
    all_entropies = []
    all_margins = []
    all_perturbations = []
    all_v2_features = []  # V2: list of 9-element feature lists
    all_v2_kl_targets = []  # V2: layer-dependent soft KL targets

    for pass_num, p_idx in enumerate(passing_indices):
        result = baseline_results[p_idx]
        problem = problems[p_idx]

        if is_master:
            print(f"\n{'─'*70}")
            print(f"[{pass_num+1}/{len(passing_indices)}] {problem.task_id} (R*={result['reward']:.2f})")

        # Forward pass
        conversation = build_code_prompt(problem)
        full_conversation = conversation + [{"role": "assistant", "content": result["generated_text"]}]
        input_ids = tokenizer.apply_chat_template(
            full_conversation, return_tensors="np", add_generation_prompt=False,
        )
        seq_len = input_ids.shape[1]
        prompt_only_ids = tokenizer.apply_chat_template(
            conversation, return_tensors="np", add_generation_prompt=True,
        )
        response_start = prompt_only_ids.shape[1]

        if is_master:
            print(f"  Seq len: {seq_len}, response: {seq_len - response_start} tokens")

        with model_mesh:
            baseline_out = elm._model(
                input_ids=jnp.array(input_ids),
                output_hidden_states=True,
            )

        baseline_logits = baseline_out.logits
        baseline_hidden = baseline_out.hidden_states
        num_layers_total = len(baseline_hidden) - 1
        valid_layers = [l for l in TARGET_LAYERS if l < num_layers_total]

        gathered_hidden = {}
        for l_idx in valid_layers:
            gathered_hidden[l_idx] = np.array(
                multihost_utils.process_allgather(baseline_hidden[l_idx + 1], tiled=True)
            )
        baseline_logits_np = np.array(
            multihost_utils.process_allgather(baseline_logits, tiled=True)
        )

        # Select response positions
        response_positions = list(range(response_start, seq_len - 1))
        if len(response_positions) > MAX_POSITIONS_PER_PROMPT:
            step = len(response_positions) / MAX_POSITIONS_PER_PROMPT
            response_positions = [response_positions[int(i * step)] for i in range(MAX_POSITIONS_PER_PROMPT)]

        if is_master:
            print(f"  {len(response_positions)} positions × {len(valid_layers)} layers")

        for pos in response_positions:
            pos_logits = baseline_logits_np[0, pos, :]
            chosen_token = int(np.argmax(pos_logits))

            # Logit features
            shifted = pos_logits - pos_logits.max()
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            sorted_logits = np.sort(pos_logits)
            margin = float(sorted_logits[-1] - sorted_logits[-2])

            # Perturbation: pick a DIFFERENT token
            masked_logits = pos_logits.copy()
            masked_logits[chosen_token] = -np.inf
            competitors = np.argsort(masked_logits)[-PERTURB_TOP_K:][::-1]
            comp_logits = masked_logits[competitors] / PERTURB_TEMP
            comp_logits -= comp_logits.max()
            comp_probs = np.exp(comp_logits) / np.sum(np.exp(comp_logits))
            rng = np.random.RandomState(int(chosen_token) * 1000 + pos)
            perturbed_token = int(competitors[rng.choice(len(comp_probs), p=comp_probs)])

            # V2: decode token string for token-type features
            token_str = tokenizer.decode([int(input_ids[0, pos])]) if V2_MODE else ""

            for li, layer_idx in enumerate(valid_layers):
                all_hidden_states.append(np.float32(gathered_hidden[layer_idx][0, pos, :]))
                all_layer_indices.append(layer_idx)
                all_positions.append(pos)
                all_entropies.append(entropy)
                all_margins.append(margin)

                # V2: compute logit probing + convergence + token-type features
                if V2_MODE:
                    prev_layer = valid_layers[li - 1] if li > 0 else None
                    v2f = _compute_v2_features(
                        gathered_hidden, pos_logits, layer_idx, prev_layer,
                        valid_layers, pos, lm_head_kernel, final_norm_weight,
                        token_str,
                    )
                    all_v2_features.append([
                        v2f["agreement"], v2f["confidence"], v2f["kl_div"],
                        v2f["int_entropy"], v2f["cos_sim"], v2f["rel_change"],
                        v2f["proj_final"], v2f["is_syntax"], v2f["is_whitespace"],
                    ])
                    # Soft KL target: 1 - exp(-kl) → 0 when converged, 1 when divergent
                    all_v2_kl_targets.append(1.0 - np.exp(-v2f["kl_div"]))

            # Build perturbed prefix for Phase 2b rollout
            perturbed_prefix_ids = np.array(input_ids[0][:pos + 1])
            perturbed_prefix_ids[pos] = perturbed_token
            prefix_response_text = tokenizer.decode(
                perturbed_prefix_ids[response_start:].tolist(),
                skip_special_tokens=True,
            )

            all_perturbations.append({
                "p_idx": int(p_idx),
                "pos": int(pos),
                "response_start": int(response_start),
                "chosen_token": int(chosen_token),
                "perturbed_token": int(perturbed_token),
                "same_token": False,
                "prefix_response_text": prefix_response_text,
                "baseline_reward": float(result["reward"]),
                "num_layers": len(valid_layers),
            })

        del baseline_out, baseline_logits, baseline_logits_np, baseline_hidden, gathered_hidden
        gc.collect()
        multihost_utils.sync_global_devices(f"phase2a_{pass_num}")

        if is_master and (pass_num + 1) % 10 == 0:
            print(f"  Progress: {pass_num+1}/{len(passing_indices)}, samples: {len(all_perturbations)}")

    multihost_utils.sync_global_devices("phase2a_passing_done")

    # ── Process failing baselines (hidden states only, regret=0) ──
    if is_master:
        print(f"\n{'─'*70}")
        print(f"Processing {len(failing_indices)} failing baselines (regret=0, no rollouts needed)")

    fail_hidden_states = []
    fail_layer_indices = []
    fail_positions = []
    fail_entropies = []
    fail_margins = []
    fail_v2_features = []
    fail_v2_kl_targets = []

    for fail_num, p_idx in enumerate(failing_indices):
        result = baseline_results[p_idx]
        problem = problems[p_idx]

        if is_master and (fail_num + 1) % 20 == 0:
            print(f"  Failing: {fail_num+1}/{len(failing_indices)}")

        conversation = build_code_prompt(problem)
        full_conversation = conversation + [{"role": "assistant", "content": result["generated_text"]}]
        input_ids = tokenizer.apply_chat_template(
            full_conversation, return_tensors="np", add_generation_prompt=False,
        )
        seq_len = input_ids.shape[1]
        prompt_only_ids = tokenizer.apply_chat_template(
            conversation, return_tensors="np", add_generation_prompt=True,
        )
        response_start = prompt_only_ids.shape[1]

        with model_mesh:
            out = elm._model(
                input_ids=jnp.array(input_ids),
                output_hidden_states=True,
            )

        logits_out = out.logits
        hidden_out = out.hidden_states
        num_layers_total = len(hidden_out) - 1
        valid_layers = [l for l in TARGET_LAYERS if l < num_layers_total]

        gathered_hidden = {}
        for l_idx in valid_layers:
            gathered_hidden[l_idx] = np.array(
                multihost_utils.process_allgather(hidden_out[l_idx + 1], tiled=True)
            )
        logits_np = np.array(
            multihost_utils.process_allgather(logits_out, tiled=True)
        )

        response_positions = list(range(response_start, seq_len - 1))
        if len(response_positions) > MAX_POSITIONS_PER_PROMPT:
            step = len(response_positions) / MAX_POSITIONS_PER_PROMPT
            response_positions = [response_positions[int(i * step)] for i in range(MAX_POSITIONS_PER_PROMPT)]

        for pos in response_positions:
            pos_logits = logits_np[0, pos, :]
            shifted = pos_logits - pos_logits.max()
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            sorted_logits = np.sort(pos_logits)
            margin = float(sorted_logits[-1] - sorted_logits[-2])

            # V2: decode token string for token-type features
            token_str = tokenizer.decode([int(input_ids[0, pos])]) if V2_MODE else ""

            for li, layer_idx in enumerate(valid_layers):
                fail_hidden_states.append(np.float32(gathered_hidden[layer_idx][0, pos, :]))
                fail_layer_indices.append(layer_idx)
                fail_positions.append(pos)
                fail_entropies.append(entropy)
                fail_margins.append(margin)

                # V2: compute features for failing baselines too
                if V2_MODE:
                    prev_layer = valid_layers[li - 1] if li > 0 else None
                    v2f = _compute_v2_features(
                        gathered_hidden, pos_logits, layer_idx, prev_layer,
                        valid_layers, pos, lm_head_kernel, final_norm_weight,
                        token_str,
                    )
                    fail_v2_features.append([
                        v2f["agreement"], v2f["confidence"], v2f["kl_div"],
                        v2f["int_entropy"], v2f["cos_sim"], v2f["rel_change"],
                        v2f["proj_final"], v2f["is_syntax"], v2f["is_whitespace"],
                    ])
                    fail_v2_kl_targets.append(1.0 - np.exp(-v2f["kl_div"]))

        del out, logits_out, logits_np, hidden_out, gathered_hidden
        gc.collect()
        multihost_utils.sync_global_devices(f"phase2a_fail_{fail_num}")

    multihost_utils.sync_global_devices("phase2a_complete")

    # Save Phase 2a data
    if is_master:
        print(f"\n  Saving Phase 2a data:")
        print(f"    Passing: {len(all_perturbations)} perturbations, {len(all_hidden_states)} hidden states")
        print(f"    Failing: {len(fail_hidden_states)} hidden states (all regret=0)")

        save_kwargs = dict(
            hidden_states=np.stack(all_hidden_states),
            layer_indices=np.array(all_layer_indices),
            positions=np.array(all_positions),
            entropies=np.array(all_entropies),
            margins=np.array(all_margins),
        )
        if V2_MODE and all_v2_features:
            save_kwargs["v2_features"] = np.array(all_v2_features, dtype=np.float32)
            save_kwargs["v2_kl_targets"] = np.array(all_v2_kl_targets, dtype=np.float32)
            print(f"    V2 features shape: {save_kwargs['v2_features'].shape}")
            print(f"    V2 KL targets: mean={np.mean(all_v2_kl_targets):.4f}, "
                  f"std={np.std(all_v2_kl_targets):.4f}")
        np.savez(PHASE2A_HIDDEN_PATH, **save_kwargs)

        with open(PHASE2A_PERTURB_PATH, "w") as f:
            json.dump(all_perturbations, f, indent=2)

        if fail_hidden_states:
            fail_kwargs = dict(
                hidden_states=np.stack(fail_hidden_states),
                layer_indices=np.array(fail_layer_indices),
                positions=np.array(fail_positions),
                entropies=np.array(fail_entropies),
                margins=np.array(fail_margins),
            )
            if V2_MODE and fail_v2_features:
                fail_kwargs["v2_features"] = np.array(fail_v2_features, dtype=np.float32)
                fail_kwargs["v2_kl_targets"] = np.array(fail_v2_kl_targets, dtype=np.float32)
            np.savez(PHASE2A_FAILING_PATH, **fail_kwargs)

        n_need_rollout = sum(1 for p in all_perturbations if not p["same_token"])
        print(f"  Saved to {PHASE2A_HIDDEN_PATH}, {PHASE2A_PERTURB_PATH}, {PHASE2A_FAILING_PATH}")
        print(f"  Perturbations needing rollout: {n_need_rollout}/{len(all_perturbations)}")

    phase2a_complete = True

if is_master:
    print(f"\n  Phase 2a {'complete' if phase2a_complete else 'skipped (Phase 2 already done)'}")


# ═════════════════════════════════════════════════════════════════════
# PHASE 2b: eSURGE ROLLOUTS — GREEDY CONTINUATION & REGRET
# ═════════════════════════════════════════════════════════════════════

# ── V2: Skip Phase 2b and assemble dataset directly from Phase 2a ──
if V2_MODE and V2_SKIP_PHASE2B and not phase2_complete and phase2a_complete:
    if is_master:
        print(f"\n{'='*70}")
        print("V2 MODE: Skipping Phase 2b rollouts — assembling dataset from logit probing targets")
        print(f"{'='*70}")

        # Load Phase 2a data
        h_data = np.load(PHASE2A_HIDDEN_PATH)
        hidden_states_all = h_data["hidden_states"]
        layer_indices_all = h_data["layer_indices"]
        positions_all = h_data["positions"]
        entropies_all = h_data["entropies"]
        margins_all = h_data["margins"]
        v2_features_all = h_data["v2_features"]
        v2_kl_targets_all = h_data["v2_kl_targets"]

        # Merge failing baseline data
        if os.path.exists(PHASE2A_FAILING_PATH):
            fail_data = np.load(PHASE2A_FAILING_PATH)
            print(f"  Merging {len(fail_data['hidden_states'])} failing baseline samples")
            hidden_states_all = np.concatenate([hidden_states_all, fail_data["hidden_states"]])
            layer_indices_all = np.concatenate([layer_indices_all, fail_data["layer_indices"]])
            positions_all = np.concatenate([positions_all, fail_data["positions"]])
            entropies_all = np.concatenate([entropies_all, fail_data["entropies"]])
            margins_all = np.concatenate([margins_all, fail_data["margins"]])
            v2_features_all = np.concatenate([v2_features_all, fail_data["v2_features"]])
            v2_kl_targets_all = np.concatenate([v2_kl_targets_all, fail_data["v2_kl_targets"]])

        # V2 uses KL targets as the primary training signal (no regret needed)
        # Store KL targets as "regret_values" for backward compatibility with Phase 3
        total = len(v2_kl_targets_all)
        print(f"\n  V2 Dataset: {total} samples")
        print(f"  KL target stats: mean={np.mean(v2_kl_targets_all):.4f}, "
              f"std={np.std(v2_kl_targets_all):.4f}, "
              f"min={np.min(v2_kl_targets_all):.4f}, max={np.max(v2_kl_targets_all):.4f}")

        # Count how many samples are "converged" (KL target < 0.1)
        converged = np.sum(v2_kl_targets_all < 0.1)
        print(f"  Converged (KL target < 0.1): {converged}/{total} ({100*converged/total:.1f}%)")

        np.savez(
            REGRET_DATA_PATH,
            hidden_states=hidden_states_all,
            layer_indices=layer_indices_all,
            positions=positions_all,
            entropies=entropies_all,
            margins=margins_all,
            regret_values=v2_kl_targets_all,  # V2: KL targets instead of regret
            v2_features=v2_features_all,
        )
        print(f"  Saved V2 dataset to {REGRET_DATA_PATH}")

    phase2_complete = True
    multihost_utils.sync_global_devices("v2_dataset_assembled")

if not phase2_complete and phase2a_complete and not (V2_MODE and V2_SKIP_PHASE2B):
    if is_master:
        print(f"\n{'='*70}")
        print("PHASE 2b: eSurge greedy rollouts for perturbed sequences")
        print(f"{'='*70}")

    # If we already loaded with Phase 2b config (direct path), skip reload.
    # Otherwise, reload model fresh (forward passes corrupt the graph).
    if not _need_phase2b_only:
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
            .set_esurge(max_model_len=512, max_num_seqs=1, hbm_utilization=0.5)
        )

    if is_master:
        print("  Building eSurge for rollouts (max_model_len=512, max_num_seqs=1, hbm_util=0.5)...")

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
    regret_values = [None] * len(perturbations)  # One per perturbation (not per layer)
    start_idx = 0

    # Load checkpoint if exists (resume from crash)
    if is_master and os.path.exists(PHASE2B_CHECKPOINT_PATH):
        try:
            with open(PHASE2B_CHECKPOINT_PATH) as f:
                ckpt = json.load(f)
            if len(ckpt["regret_values"]) == len(perturbations):
                start_idx = ckpt["completed"]
                for i in range(start_idx):
                    regret_values[i] = ckpt["regret_values"][i]
                print(f"  Resuming from checkpoint: {start_idx}/{len(perturbations)} already done")
            else:
                print(f"  Checkpoint size mismatch ({len(ckpt['regret_values'])} vs {len(perturbations)}), starting fresh")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}, starting fresh")

    # Broadcast start_idx to all workers
    start_arr = multihost_utils.broadcast_one_to_all(jnp.array([start_idx]))
    start_idx = int(start_arr[0])

    t_rollout_start = time.time()

    for pert_idx in range(start_idx, len(perturbations)):
        pert = perturbations[pert_idx]
        if pert["same_token"]:
            regret_values[pert_idx] = 0.0
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
            perturbed_reward, _ = execute_code(perturbed_code, problem.test_code, timeout=CODE_TIMEOUT, partial=True)

        regret = max(0.0, pert["baseline_reward"] - perturbed_reward)
        regret_values[pert_idx] = regret

        if is_master and (pert_idx + 1) % 50 == 0:
            elapsed = time.time() - t_rollout_start
            done_so_far = [v for v in regret_values if v is not None]
            nonzero = sum(1 for v in done_so_far if v > 0)
            print(f"  [{pert_idx+1}/{len(perturbations)}] {elapsed:.0f}s, "
                  f"non-zero regret: {nonzero}/{len(done_so_far)} ({100*nonzero/max(len(done_so_far),1):.1f}%)")

        # Periodic checkpoint save (master only)
        if is_master and (pert_idx + 1) % PHASE2B_CHECKPOINT_INTERVAL == 0:
            ckpt_data = {
                "completed": pert_idx + 1,
                "regret_values": [v if v is not None else 0.0 for v in regret_values],
            }
            with open(PHASE2B_CHECKPOINT_PATH, "w") as f:
                json.dump(ckpt_data, f)
            print(f"  [Checkpoint saved: {pert_idx+1}/{len(perturbations)}]")

    # Finalize: replace any None values with 0.0
    regret_values = [v if v is not None else 0.0 for v in regret_values]

    # Save final checkpoint
    if is_master:
        ckpt_data = {
            "completed": len(perturbations),
            "regret_values": regret_values,
        }
        with open(PHASE2B_CHECKPOINT_PATH, "w") as f:
            json.dump(ckpt_data, f)

    # Broadcast regret values from master
    r_arr = jnp.array(regret_values if is_master else [0.0] * len(regret_values))
    r_arr = multihost_utils.broadcast_one_to_all(r_arr)
    regret_values = r_arr.tolist()

    multihost_utils.sync_global_devices("phase2b_complete")

    # Clean up eSurge
    del esurge
    gc.collect()

    # Assemble final regret dataset: expand regret per position → per layer × position
    # Then merge with failing baseline data (all regret=0)
    if is_master:
        elapsed_total = time.time() - t_rollout_start
        print(f"\n  Rollouts complete in {elapsed_total:.0f}s")

        # Load Phase 2a hidden states (passing baselines)
        h_data = np.load(PHASE2A_HIDDEN_PATH)
        hidden_states_all = h_data["hidden_states"]
        layer_indices_all = h_data["layer_indices"]
        positions_all = h_data["positions"]
        entropies_all = h_data["entropies"]
        margins_all = h_data["margins"]

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

        # V2: load V2 features from Phase 2a if available
        v2_features_all = None
        if V2_MODE and "v2_features" in h_data:
            v2_features_all = h_data["v2_features"]

        # Merge failing baseline data (all regret=0, no rollouts needed)
        if os.path.exists(PHASE2A_FAILING_PATH):
            fail_data = np.load(PHASE2A_FAILING_PATH)
            fail_hs = fail_data["hidden_states"]
            fail_li = fail_data["layer_indices"]
            fail_pos = fail_data["positions"]
            fail_ent = fail_data["entropies"]
            fail_mar = fail_data["margins"]
            fail_regret = np.zeros(len(fail_hs))

            print(f"\n  Merging {len(fail_hs)} failing baseline samples (regret=0)")
            hidden_states_all = np.concatenate([hidden_states_all, fail_hs])
            layer_indices_all = np.concatenate([layer_indices_all, fail_li])
            positions_all = np.concatenate([positions_all, fail_pos])
            entropies_all = np.concatenate([entropies_all, fail_ent])
            margins_all = np.concatenate([margins_all, fail_mar])
            expanded_regret = np.concatenate([expanded_regret, fail_regret])

            if v2_features_all is not None and "v2_features" in fail_data:
                v2_features_all = np.concatenate([v2_features_all, fail_data["v2_features"]])

        # Save final regret dataset
        total = len(expanded_regret)
        print(f"\n{'='*70}")
        print(f"Saving regret dataset: {total} samples")

        save_kwargs = dict(
            hidden_states=hidden_states_all,
            layer_indices=layer_indices_all,
            positions=positions_all,
            entropies=entropies_all,
            margins=margins_all,
            regret_values=expanded_regret,
        )
        if v2_features_all is not None:
            save_kwargs["v2_features"] = v2_features_all
        np.savez(REGRET_DATA_PATH, **save_kwargs)
        print(f"  Saved to {REGRET_DATA_PATH}")

        nonzero_count = np.sum(expanded_regret > 0)
        print(f"  Non-zero regret: {nonzero_count}/{total} ({100*nonzero_count/total:.1f}%)")
        print(f"  Zero regret: {total - nonzero_count}/{total} ({100*(total-nonzero_count)/total:.1f}%)")
        print(f"  Mean regret: {np.mean(expanded_regret):.4f}")
        print(f"  Max regret: {np.max(expanded_regret):.4f}")

        # Clean up Phase 2b checkpoint now that final dataset is saved
        if os.path.exists(PHASE2B_CHECKPOINT_PATH):
            os.remove(PHASE2B_CHECKPOINT_PATH)
            print(f"  Cleaned up {PHASE2B_CHECKPOINT_PATH}")


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

    # Load logit features (entropy, margin)
    if "entropies" in data:
        entropies_raw = jnp.array(data["entropies"], dtype=jnp.float32)
        margins_raw = jnp.array(data["margins"], dtype=jnp.float32)
        has_logit_features = True
    else:
        has_logit_features = False
        entropies_raw = jnp.zeros(len(regret_targets))
        margins_raw = jnp.zeros(len(regret_targets))

    # V2: load extra features if available
    has_v2_features = "v2_features" in data
    if has_v2_features:
        v2_extra_raw = jnp.array(data["v2_features"], dtype=jnp.float32)
        num_meta = 4 + v2_extra_raw.shape[1]  # base (4) + V2 extras
        print(f"  V2 features detected: {v2_extra_raw.shape[1]} extra features (total meta={num_meta})")
    else:
        v2_extra_raw = None
        num_meta = 4

    N = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    print(f"  Dataset: {N} samples, hidden_dim={hidden_dim}")
    if has_logit_features:
        print(f"  Logit features: entropy range [{float(jnp.min(entropies_raw)):.2f}, {float(jnp.max(entropies_raw)):.2f}], "
              f"margin range [{float(jnp.min(margins_raw)):.2f}, {float(jnp.max(margins_raw)):.2f}]")

    # V2: use soft KL targets directly (continuous [0,1], already layer-dependent)
    # Non-V2: binary targets with label smoothing
    if has_v2_features:
        # regret_targets are actually soft KL targets in V2 mode (stored as regret_values)
        # Threshold for binary classification: converged (< 0.5) vs not converged (>= 0.5)
        binary_raw = (regret_targets > 0.5).astype(jnp.float32)
        # Use the continuous KL target directly (no label smoothing needed — already soft)
        training_targets = regret_targets
        nonzero_frac = float(jnp.mean(binary_raw))
        print(f"  V2 KL targets: mean={float(jnp.mean(regret_targets)):.4f}, "
              f"std={float(jnp.std(regret_targets)):.4f}")
        print(f"  Not converged (>0.5): {nonzero_frac:.3f} ({int(jnp.sum(binary_raw))}/{N})")
    else:
        binary_raw = (regret_targets > 0).astype(jnp.float32)
        training_targets = binary_raw * (1 - LABEL_SMOOTHING) + (1 - binary_raw) * LABEL_SMOOTHING
        nonzero_frac = float(jnp.mean(binary_raw))
        print(f"  Fragile fraction: {nonzero_frac:.3f} ({int(jnp.sum(binary_raw))}/{N})")
        print(f"  Label smoothing: {LABEL_SMOOTHING} (targets: 0→{LABEL_SMOOTHING}, 1→{1-LABEL_SMOOTHING})")

    # ── Per-layer hidden state normalization ──
    norm_stats = {}
    unique_layers = sorted(set(int(x) for x in np.unique(np.array(layer_indices))))
    print(f"  Computing per-layer normalization stats...")
    for layer in unique_layers:
        mask = np.array(layer_indices) == layer
        layer_hs = np.array(hidden_states[mask])
        mean = np.mean(layer_hs, axis=0)
        std = np.std(layer_hs, axis=0)
        norm_stats[f"layer_{layer}_mean"] = mean
        norm_stats[f"layer_{layer}_std"] = std
        print(f"    Layer {layer}: mean_norm={np.linalg.norm(mean):.2f}, "
              f"mean_std={np.mean(std):.4f}, min_std={np.min(std):.6f}")
        hidden_states = hidden_states.at[mask].set(
            (hidden_states[mask] - jnp.array(mean)) / (jnp.array(std) + 1e-8)
        )

    # Normalize entropy and margin (z-score)
    ent_mean = float(jnp.mean(entropies_raw))
    ent_std = float(jnp.std(entropies_raw))
    mar_mean = float(jnp.mean(margins_raw))
    mar_std = float(jnp.std(margins_raw))
    norm_stats["entropy_mean"] = ent_mean
    norm_stats["entropy_std"] = ent_std
    norm_stats["margin_mean"] = mar_mean
    norm_stats["margin_std"] = mar_std
    entropies_norm = ((entropies_raw - ent_mean) / (ent_std + 1e-8)).reshape(-1, 1)
    margins_norm = ((margins_raw - mar_mean) / (mar_std + 1e-8)).reshape(-1, 1)
    if has_logit_features:
        print(f"  Entropy: mean={ent_mean:.3f}, std={ent_std:.3f}")
        print(f"  Margin:  mean={mar_mean:.3f}, std={mar_std:.3f}")

    # V2: normalize extra features (z-score per feature)
    v2_extra_norm = None
    if has_v2_features:
        v2_feature_names = ["agreement", "confidence", "kl_div", "int_entropy",
                            "cos_sim", "rel_change", "proj_final",
                            "is_syntax", "is_whitespace"]
        v2_extra_norm = v2_extra_raw.copy()
        for i, fname in enumerate(v2_feature_names):
            col = v2_extra_raw[:, i]
            col_mean = float(jnp.mean(col))
            col_std = float(jnp.std(col))
            norm_stats[f"v2_{fname}_mean"] = col_mean
            norm_stats[f"v2_{fname}_std"] = col_std
            if col_std > 1e-6:
                v2_extra_norm = v2_extra_norm.at[:, i].set((col - col_mean) / (col_std + 1e-8))
            print(f"    V2 {fname}: mean={col_mean:.4f}, std={col_std:.4f}")

    # Normalize layer index and position to [0, 1]
    num_layers = max(int(jnp.max(layer_indices)) + 1, 1)
    max_pos = max(int(jnp.max(positions)) + 1, 1)
    layer_norm = layer_indices.astype(jnp.float32).reshape(-1, 1) / (num_layers - 1)
    pos_norm = positions.astype(jnp.float32).reshape(-1, 1) / (max_pos - 1)

    # Stratified train/val split: preserve fragile/safe ratio in both sets
    perm = np.random.RandomState(42).permutation(N)
    fragile_idx = perm[np.array(training_targets[perm]) > (0.5 if has_v2_features else 0)]
    safe_idx = perm[np.array(training_targets[perm]) <= (0.5 if has_v2_features else 0)]
    f_split = int(len(fragile_idx) * TRAIN_SPLIT)
    s_split = int(len(safe_idx) * TRAIN_SPLIT)
    train_idx = np.concatenate([fragile_idx[:f_split], safe_idx[:s_split]])
    val_idx = np.concatenate([fragile_idx[f_split:], safe_idx[s_split:]])
    np.random.RandomState(42).shuffle(train_idx)
    np.random.RandomState(42).shuffle(val_idx)

    print(f"  Train: {len(train_idx)} (fragile: {f_split}), Val: {len(val_idx)} (fragile: {len(fragile_idx)-f_split})")

    # Create model and optimizer
    model = create_estimator(hidden_dim, seed=42, dropout_rate=ESTIMATOR_DROPOUT,
                             num_meta_features=num_meta)

    # Cosine LR schedule with warmup
    warmup_steps = 50
    total_steps = ESTIMATOR_EPOCHS * (len(train_idx) // ESTIMATOR_BATCH + 1)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, ESTIMATOR_LR, warmup_steps),
            optax.cosine_decay_schedule(ESTIMATOR_LR, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=ESTIMATOR_WEIGHT_DECAY), wrt=nnx.Param)

    # Positive weight for BCE loss (balance classes)
    pos_weight = (1.0 - nonzero_frac) / max(nonzero_frac, 1e-6)
    print(f"  BCE pos_weight: {pos_weight:.3f} (balancing {nonzero_frac:.1%} positives)")

    # Layer weights: earlier layers get higher weight to improve early-exit predictions
    max_target_layer = max(TARGET_LAYERS)
    min_target_layer = min(TARGET_LAYERS)
    layer_weight_map = {}
    for tl in TARGET_LAYERS:
        frac = (tl - min_target_layer) / max(max_target_layer - min_target_layer, 1)
        layer_weight_map[tl] = 2.0 - frac
    print(f"  Layer weights: {min_target_layer}→{layer_weight_map[min_target_layer]:.1f}, "
          f"{max_target_layer}→{layer_weight_map[max_target_layer]:.1f}")

    # Pre-compute per-sample layer weights
    sample_layer_weights = jnp.ones(N)
    for tl, w in layer_weight_map.items():
        mask = layer_indices == tl
        sample_layer_weights = sample_layer_weights.at[mask].set(w)

    def loss_fn(model, h, l, p, ent, mar, ef, targets, weights):
        preds = model(h, l, p, ent, mar, extra_features=ef, deterministic=False).squeeze(-1)
        eps = 1e-7
        preds = jnp.clip(preds, eps, 1 - eps)
        bce = -(pos_weight * targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
        return jnp.mean(bce * weights)

    def train_step(model, optimizer, h, l, p, ent, mar, ef, targets, weights):
        loss, grads = nnx.value_and_grad(loss_fn)(model, h, l, p, ent, mar, ef, targets, weights)
        optimizer.update(model, grads)
        return loss

    print(f"  Training for up to {ESTIMATOR_EPOCHS} epochs (patience={ESTIMATOR_PATIENCE})")
    print(f"  LR: cosine decay from {ESTIMATOR_LR} with {warmup_steps}-step warmup")
    print(f"  Dropout: {ESTIMATOR_DROPOUT}, Weight decay: {ESTIMATOR_WEIGHT_DECAY}")
    if has_v2_features:
        print(f"  V2 MODE: training on soft KL targets with {num_meta} metadata features")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(ESTIMATOR_EPOCHS):
        epoch_perm = np.random.RandomState(epoch).permutation(len(train_idx))
        shuffled = train_idx[epoch_perm]

        epoch_losses = []
        for batch_start in range(0, len(shuffled), ESTIMATOR_BATCH):
            batch_end = min(batch_start + ESTIMATOR_BATCH, len(shuffled))
            idx = shuffled[batch_start:batch_end]

            ef_batch = v2_extra_norm[idx] if v2_extra_norm is not None else None
            loss = train_step(
                model, optimizer,
                hidden_states[idx], layer_norm[idx], pos_norm[idx],
                entropies_norm[idx], margins_norm[idx],
                ef_batch,
                training_targets[idx], sample_layer_weights[idx],
            )
            epoch_losses.append(float(loss))

        train_loss = np.mean(epoch_losses)

        # Validation: BCE loss + accuracy (deterministic mode = no dropout)
        ef_val = v2_extra_norm[val_idx] if v2_extra_norm is not None else None
        val_preds = model(hidden_states[val_idx], layer_norm[val_idx], pos_norm[val_idx],
                          entropies_norm[val_idx], margins_norm[val_idx],
                          extra_features=ef_val,
                          deterministic=True).squeeze(-1)
        val_bce = float(jnp.mean(
            -(training_targets[val_idx] * jnp.log(jnp.clip(val_preds, 1e-7, 1-1e-7))
              + (1 - training_targets[val_idx]) * jnp.log(jnp.clip(1 - val_preds, 1e-7, 1-1e-7)))
        ))
        val_acc = float(jnp.mean((val_preds > 0.5) == binary_raw[val_idx]))

        if val_bce < best_val_loss:
            best_val_loss = val_bce
            best_val_acc = val_acc
            patience_counter = 0
            save_estimator(model, ESTIMATOR_PATH, norm_stats=norm_stats)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter == 0:
            print(f"  Epoch {epoch+1:3d}: train_bce={train_loss:.6f}, val_bce={val_bce:.6f}, "
                  f"val_acc={val_acc:.3f}{' *best*' if patience_counter == 0 else ''}")

        if patience_counter >= ESTIMATOR_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {ESTIMATOR_PATIENCE} epochs)")
            break

    print(f"\n  Training complete. Best val BCE: {best_val_loss:.6f}, Best val acc: {best_val_acc:.3f}")
    print(f"  Model saved to {ESTIMATOR_PATH}.npz (with per-layer norm stats)")

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
