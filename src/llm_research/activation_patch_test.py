"""
Activation Patching Test
========================
Generates text from the model, modifies hidden-state activations at a
chosen intermediate layer, re-runs the remaining transformer layers to
get altered logits/tokens, and optionally continues generation from the
patched prefix to observe how the output diverges.

Based on the working multi-host patterns from elarge_test.py.
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

# ── CONFIGURABLE PARAMETERS ────────────────────────────────────────────
TARGET_LAYER        = 20       # Which layer's activations to patch (0-indexed)
PATCH_SCALE         = 2.0      # Scale factor for perturbation (higher = bigger change)
PATCH_START_POS     = -20      # Start of token positions to patch (negative = from end)
PATCH_END_POS       = -1       # End of token positions to patch (negative = from end)
COMPARE_LAST_N      = 10       # How many final tokens to compare original vs patched
MAX_GEN_TOKENS      = 256      # Max tokens for initial generation
CONTINUE_GEN_TOKENS = 128      # Max tokens for continued generation from patched prefix

PROMPTS = [
    "Explain the difference between TCP and UDP in one paragraph.",
    "Write a Python function that checks whether a string is a valid palindrome, ignoring spaces and punctuation.",
    "What is the ship of Theseus paradox and what does it tell us about identity?",
]

# ── 1. INITIALIZE DISTRIBUTED SYSTEM ───────────────────────────────────
jax.distributed.initialize()
from jax.experimental import multihost_utils
import easydel as ed
from transformers import AutoTokenizer

is_master = jax.process_index() == 0

if is_master:
    print(f"╔{'═'*68}╗")
    print(f"║{'ACTIVATION PATCHING TEST':^68}║")
    print(f"╠{'═'*68}╣")
    print(f"║ Total devices: {jax.device_count():<52}║")
    print(f"║ Local devices: {jax.local_device_count():<52}║")
    print(f"║ Target layer:  {TARGET_LAYER:<52}║")
    print(f"║ Patch scale:   {PATCH_SCALE:<52}║")
    print(f"║ Patch range:   [{PATCH_START_POS}, {PATCH_END_POS}] (relative to seq end){' '*(52 - len(f'[{PATCH_START_POS}, {PATCH_END_POS}] (relative to seq end)'))}║")
    print(f"║ Prompts:       {len(PROMPTS):<52}║")
    print(f"╚{'═'*68}╝")

# ── 2. LOAD MODEL ──────────────────────────────────────────────────────
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

axis_dims = (1, 1, 8, 4, 1)
axis_names = ("dp", "fsdp", "tp", "sp", "selective")

elm = (
    ed.eLargeModel.from_pretrained(model_id)
    .set_dtype("bf16")
    .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
    .set_esurge(max_model_len=4096, max_num_seqs=4)
)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: GENERATE RESPONSES
# ═══════════════════════════════════════════════════════════════════════
if is_master:
    print("\n" + "=" * 70)
    print("PHASE 1: Generating baseline responses with eSurge")
    print("=" * 70)

esurge = elm.build_esurge()

generated_responses = []

for prompt_idx, prompt in enumerate(PROMPTS):
    if is_master:
        print(f"\n{'─'*70}")
        print(f"[{prompt_idx + 1}/{len(PROMPTS)}] {prompt}")
        print(f"{'─'*70}")
        print("Response: ", end="", flush=True)

    conversation = [{"role": "user", "content": prompt}]
    generated_text = ""
    num_tokens = 0
    t0 = time.time()

    for output in esurge.chat(
        conversation,
        sampling_params=ed.SamplingParams(max_tokens=MAX_GEN_TOKENS),
        stream=True,
    ):
        generated_text += output.delta_text
        num_tokens += 1
        if is_master:
            print(output.delta_text, end="", flush=True)

    elapsed = time.time() - t0
    tps = num_tokens / elapsed if elapsed > 0 else 0.0

    if is_master:
        print(f"\n  [{num_tokens} tokens in {elapsed:.1f}s — {tps:.1f} tok/s]")

    generated_responses.append({
        "prompt": prompt,
        "conversation": conversation,
        "generated_text": generated_text,
        "num_tokens": num_tokens,
        "elapsed": elapsed,
        "tps": tps,
    })

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: SHUTDOWN eSurge, PREPARE FOR FORWARD PASSES
# ═══════════════════════════════════════════════════════════════════════
if is_master:
    print("\n" + "=" * 70)
    print("PHASE 2: Shutting down eSurge, preparing for layer-by-layer passes")
    print("=" * 70)

del esurge
gc.collect()
multihost_utils.sync_global_devices("esurge_stopped")

# Monkey-patch EasyDeL output dataclass validation for multi-host
import easydel.infra.modeling_outputs as _mo
for _name, _cls in inspect.getmembers(_mo, inspect.isclass):
    if hasattr(_cls, '__post_init__'):
        _cls.__post_init__ = lambda self: None

if is_master:
    print("  ✓ Patched output validation for multi-host")

model_mesh = elm._model.config.mesh

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: DISCOVER MODEL STRUCTURE
# ═══════════════════════════════════════════════════════════════════════
if is_master:
    print("\n" + "=" * 70)
    print("PHASE 3: Discovering model internals")
    print("=" * 70)

# Print the model structure so we know exactly what attributes exist
model = elm._model
if is_master:
    print(f"  Model type:   {type(model).__name__}")
    print(f"  Top-level attrs: {[a for a in dir(model) if not a.startswith('_')][:30]}")

    # Find the base model (typically .model)
    if hasattr(model, 'model'):
        base = model.model
        print(f"  Base model type: {type(base).__name__}")
        print(f"  Base attrs:      {[a for a in dir(base) if not a.startswith('_')][:30]}")

        if hasattr(base, 'layers'):
            layers = base.layers
            num_layers = len(layers) if hasattr(layers, '__len__') else "unknown"
            print(f"  Number of layers: {num_layers}")
            if num_layers != "unknown" and num_layers > 0:
                layer0 = layers[0]
                print(f"  Layer type:      {type(layer0).__name__}")
                # Inspect __call__ signature
                if hasattr(layer0, '__call__'):
                    sig = inspect.signature(layer0.__call__)
                    print(f"  Layer __call__ sig: {sig}")
        else:
            print("  ⚠ No 'layers' attribute found on base model")

        if hasattr(base, 'norm'):
            print(f"  Norm type:       {type(base.norm).__name__}")
        else:
            print("  ⚠ No 'norm' attribute found on base model")

    if hasattr(model, 'lm_head'):
        print(f"  lm_head type:    {type(model.lm_head).__name__}")
    else:
        print("  ⚠ No 'lm_head' attribute found on model")

multihost_utils.sync_global_devices("discovery_done")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: BASELINE + PATCHED FORWARD PASSES
# ═══════════════════════════════════════════════════════════════════════
if is_master:
    print("\n" + "=" * 70)
    print("PHASE 4: Baseline & patched forward passes")
    print("=" * 70)

all_results = []

for r_idx, resp in enumerate(generated_responses):
    prompt = resp["prompt"]
    conversation = resp["conversation"]
    generated_text = resp["generated_text"]

    if is_master:
        print(f"\n{'━'*70}")
        print(f"[{r_idx + 1}/{len(PROMPTS)}] {prompt[:60]}...")
        print(f"{'━'*70}")

    # ── Tokenize the full conversation (prompt + response) ──
    full_conversation = conversation + [{"role": "assistant", "content": generated_text}]
    input_ids = tokenizer.apply_chat_template(
        full_conversation,
        return_tensors="np",
        add_generation_prompt=False,
    )
    seq_len = input_ids.shape[1]

    if is_master:
        print(f"  Sequence length: {seq_len}")

    # ── Resolve patch positions from negative indices ──
    abs_start = seq_len + PATCH_START_POS if PATCH_START_POS < 0 else PATCH_START_POS
    abs_end   = seq_len + PATCH_END_POS   if PATCH_END_POS   < 0 else PATCH_END_POS
    abs_start = max(0, abs_start)
    abs_end   = min(seq_len - 1, abs_end)

    if is_master:
        print(f"  Patching positions: [{abs_start}, {abs_end}] ({abs_end - abs_start + 1} tokens)")

    # ── 4a. BASELINE FORWARD PASS (full model) ──
    if is_master:
        print("  Running baseline forward pass...")

    with model_mesh:
        baseline_out = elm._model(
            input_ids=jnp.array(input_ids),
            output_hidden_states=True,
        )

    baseline_logits = baseline_out.logits           # (1, seq_len, vocab)
    baseline_hidden = baseline_out.hidden_states     # tuple of (1, seq_len, hidden_dim) per layer
    num_layers_total = len(baseline_hidden) - 1      # hidden_states[0] = embeddings, [1..N] = layer outputs

    if is_master:
        print(f"  Total transformer layers: {num_layers_total}")

    # Validate target layer
    actual_target = min(TARGET_LAYER, num_layers_total - 1)
    if actual_target != TARGET_LAYER and is_master:
        print(f"  ⚠ TARGET_LAYER {TARGET_LAYER} > max {num_layers_total-1}, clamped to {actual_target}")

    if is_master:
        print(f"  Using target layer: {actual_target}")

    # ── 4b. PATCH ACTIVATIONS & RE-RUN REMAINING LAYERS ──
    if is_master:
        print("  Patching activations and re-running remaining layers...")

    with model_mesh:
        # Get the hidden state output FROM the target layer
        # hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1
        # So hidden_states[actual_target + 1] = output of decoder layer `actual_target`
        patched_hidden = baseline_hidden[actual_target + 1]  # (1, seq_len, hidden_dim)

        # Generate deterministic noise — SAME across all workers to prevent desync
        rng = jax.random.PRNGKey(42)
        noise = jax.random.normal(rng, shape=(abs_end - abs_start + 1, patched_hidden.shape[-1]))

        # Scale noise relative to the activation magnitudes in the patch region
        patch_region = patched_hidden[0, abs_start:abs_end + 1, :]  # (patch_len, hidden_dim)
        region_std = jnp.std(patch_region)
        scaled_noise = noise * region_std * PATCH_SCALE

        if is_master:
            print(f"  Activation std in patch region: {float(region_std):.6f}")
            print(f"  Noise magnitude (after scaling): {float(jnp.std(scaled_noise)):.6f}")

        # Apply the perturbation
        patched_hidden = patched_hidden.at[0, abs_start:abs_end + 1, :].add(scaled_noise)

        # ── Run remaining layers manually ──
        # We need to pass the patched hidden states through layers [actual_target+1 ... N-1]
        h = patched_hidden
        remaining_layers = elm._model.model.layers[actual_target + 1:]

        if is_master:
            print(f"  Running {len(remaining_layers)} remaining layers...")

        # Build causal attention mask and position IDs for the full sequence
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
        position_ids = jnp.arange(seq_len)[None, :]  # (1, seq_len)

        for layer_idx, layer in enumerate(remaining_layers):
            global_layer_idx = actual_target + 1 + layer_idx
            try:
                # Try the most common Flax NNX layer call signature
                layer_out = layer(
                    h,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                )
                # Handle tuple output (hidden_states, ...) or single tensor
                if isinstance(layer_out, tuple):
                    h = layer_out[0]
                else:
                    h = layer_out
            except TypeError as e:
                # If the signature doesn't match, try without attention_mask
                if is_master:
                    print(f"  ⚠ Layer {global_layer_idx} call failed with: {e}")
                    print(f"    Trying minimal call signature...")
                try:
                    layer_out = layer(h)
                    if isinstance(layer_out, tuple):
                        h = layer_out[0]
                    else:
                        h = layer_out
                except Exception as e2:
                    if is_master:
                        print(f"    ✗ Minimal call also failed: {e2}")
                        sig = inspect.signature(layer.__call__)
                        print(f"    Layer signature: {sig}")
                    # Can't continue — break and report
                    break

        # Apply final norm + lm_head
        if is_master:
            print("  Applying final norm + lm_head...")

        patched_normed = elm._model.model.norm(h)
        patched_logits = elm._model.lm_head(patched_normed)  # (1, seq_len, vocab)

    # ── 4c. COMPARE ORIGINAL vs PATCHED ──
    if is_master:
        print(f"\n  ┌─ COMPARISON (last {COMPARE_LAST_N} positions) ─────────────────────────────")

    compare_start = max(0, seq_len - COMPARE_LAST_N)

    baseline_probs = jax.nn.softmax(baseline_logits, axis=-1)
    patched_probs  = jax.nn.softmax(patched_logits, axis=-1)

    # Per-position KL divergence: KL(baseline || patched)
    kl_divs = []
    token_comparisons = []

    for pos in range(compare_start, seq_len):
        bp = baseline_probs[0, pos, :]
        pp = patched_probs[0, pos, :]

        # KL divergence (clamp for numerical stability)
        kl = float(jnp.sum(bp * jnp.log(jnp.clip(bp, 1e-10, 1.0) / jnp.clip(pp, 1e-10, 1.0))))
        kl_divs.append(kl)

        baseline_token_id = int(jnp.argmax(bp))
        patched_token_id  = int(jnp.argmax(pp))

        baseline_token_str = tokenizer.decode([baseline_token_id])
        patched_token_str  = tokenizer.decode([patched_token_id])

        # Top-3 for each
        b_top3_idx = jnp.argsort(bp)[-3:][::-1]
        p_top3_idx = jnp.argsort(pp)[-3:][::-1]
        b_top3 = [(tokenizer.decode([int(b_top3_idx[k])]), float(bp[b_top3_idx[k]])) for k in range(3)]
        p_top3 = [(tokenizer.decode([int(p_top3_idx[k])]), float(pp[p_top3_idx[k]])) for k in range(3)]

        changed = "≠" if baseline_token_id != patched_token_id else "="
        token_comparisons.append({
            "pos": pos,
            "original": baseline_token_str,
            "patched": patched_token_str,
            "changed": changed != "=",
            "kl_div": kl,
            "original_top3": b_top3,
            "patched_top3": p_top3,
        })

        if is_master:
            in_patch = "🔧" if abs_start <= pos <= abs_end else "  "
            print(f"  │ {in_patch} pos {pos:4d}: '{baseline_token_str}' {changed} '{patched_token_str}' "
                  f"  KL={kl:.4f}")

    avg_kl = np.mean(kl_divs) if kl_divs else 0.0
    num_changed = sum(1 for tc in token_comparisons if tc["changed"])

    if is_master:
        print(f"  └{'─'*68}")
        print(f"  Summary: {num_changed}/{len(token_comparisons)} tokens changed, avg KL divergence = {avg_kl:.6f}")

    # ── 4d. DECODE FULL PATCHED SEQUENCE ──
    patched_token_ids_all = jnp.argmax(patched_logits[0], axis=-1)  # (seq_len,)
    # The logits at position i predict position i+1, so shift
    patched_decoded_ids = patched_token_ids_all[:-1]  # predictions for positions 1..seq_len-1

    # Decode the response portion only (everything after the prompt)
    # Find where the original prompt tokens end
    prompt_only_ids = tokenizer.apply_chat_template(
        conversation,
        return_tensors="np",
        add_generation_prompt=True,
    )
    prompt_len = prompt_only_ids.shape[1]

    original_response_ids = input_ids[0, prompt_len:]
    patched_response_ids  = patched_decoded_ids[prompt_len - 1:]  # shifted by 1 since logits predict next

    original_response_text = tokenizer.decode(original_response_ids.tolist(), skip_special_tokens=True)
    patched_response_text  = tokenizer.decode(patched_response_ids.tolist(), skip_special_tokens=True)

    if is_master:
        print(f"\n  ┌─ ORIGINAL RESPONSE {'─'*48}")
        for line in original_response_text[:500].split('\n'):
            print(f"  │ {line}")
        print(f"  ├─ PATCHED RESPONSE (greedy from altered logits) {'─'*20}")
        for line in patched_response_text[:500].split('\n'):
            print(f"  │ {line}")
        print(f"  └{'─'*68}")

    # Store results
    result = {
        "prompt_index": r_idx,
        "prompt": prompt,
        "original_response": generated_text,
        "patched_response_decoded": patched_response_text[:1000],
        "target_layer": actual_target,
        "patch_scale": PATCH_SCALE,
        "patch_positions": [abs_start, abs_end],
        "seq_len": int(seq_len),
        "num_layers": int(num_layers_total),
        "tokens_changed": num_changed,
        "tokens_compared": len(token_comparisons),
        "avg_kl_divergence": float(avg_kl),
        "token_comparisons": token_comparisons,
    }
    all_results.append(result)

    # Free memory before next prompt
    del baseline_out, baseline_logits, baseline_hidden, baseline_probs
    del patched_hidden, patched_logits, patched_probs, patched_normed, h
    gc.collect()
    multihost_utils.sync_global_devices(f"patch_done_{r_idx}")

    if is_master:
        print(f"  ✓ Prompt {r_idx + 1}/{len(PROMPTS)} complete")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: CONTINUED GENERATION FROM PATCHED PREFIX
# ═══════════════════════════════════════════════════════════════════════
if is_master:
    print("\n" + "=" * 70)
    print("PHASE 5: Continued generation from patched prefixes")
    print("=" * 70)

# Rebuild eSurge for continued generation
esurge2 = elm.build_esurge()

for r_idx, resp in enumerate(generated_responses):
    result = all_results[r_idx]
    prompt = resp["prompt"]
    conversation = resp["conversation"]
    original_text = resp["generated_text"]

    # Take the patched response text and use it as a partial prefix to
    # continue generating from
    patched_prefix = result["patched_response_decoded"]

    # Trim to first ~half of original response length to leave room for divergence
    prefix_tokens = tokenizer.encode(patched_prefix)
    half_len = len(prefix_tokens) // 2
    if half_len > 10:
        trimmed_prefix = tokenizer.decode(prefix_tokens[:half_len], skip_special_tokens=True)
    else:
        trimmed_prefix = patched_prefix[:200]

    if is_master:
        print(f"\n{'─'*70}")
        print(f"[{r_idx + 1}/{len(PROMPTS)}] {prompt[:60]}...")
        print(f"  Feeding patched prefix ({len(trimmed_prefix)} chars) back for continued generation")
        print(f"{'─'*70}")

    # Build a conversation where the assistant has already started responding
    # with the patched prefix, then let the model continue
    patched_conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": trimmed_prefix},
    ]

    # Tokenize as a prefix for continued generation
    continued_input = tokenizer.apply_chat_template(
        patched_conversation,
        return_tensors="np",
        add_generation_prompt=False,
    )

    # For continued generation, we present this as a new chat with
    # the partially-completed assistant message. eSurge's chat() adds
    # the generation prompt automatically, so we use a trick: send the
    # patched prefix as part of the user message context.
    continuation_prompt = (
        f"{prompt}\n\n"
        f"[Continue from this partial response]: {trimmed_prefix}"
    )
    continue_conversation = [{"role": "user", "content": continuation_prompt}]

    continued_text = ""
    if is_master:
        print("  Continued: ", end="", flush=True)

    for output in esurge2.chat(
        continue_conversation,
        sampling_params=ed.SamplingParams(max_tokens=CONTINUE_GEN_TOKENS),
        stream=True,
    ):
        continued_text += output.delta_text
        if is_master:
            print(output.delta_text, end="", flush=True)

    if is_master:
        print()

    result["continued_from_patch"] = continued_text
    result["patched_prefix_used"] = trimmed_prefix

    if is_master:
        print(f"\n  ┌─ ORIGINAL FULL RESPONSE (first 300 chars) {'─'*24}")
        for line in original_text[:300].split('\n'):
            print(f"  │ {line}")
        print(f"  ├─ PATCHED CONTINUATION {'─'*43}")
        for line in continued_text[:300].split('\n'):
            print(f"  │ {line}")
        print(f"  └{'─'*68}")

# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
if is_master:
    output_path = "activation_patch_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

# ═══════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════
if is_master:
    print("\nCleaning up...")

del esurge2
del elm
gc.collect()
multihost_utils.sync_global_devices("ready_to_kill")
jax.distributed.shutdown()
sys.exit(0)
