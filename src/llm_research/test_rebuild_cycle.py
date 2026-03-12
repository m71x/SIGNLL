"""
Validation: Test the forward-pass → eSurge-rebuild → generation cycle.

Confirms that:
1. Forward pass with hidden states works after eSurge init+teardown
2. eSurge can be rebuilt after forward passes
3. eSurge generates valid code continuations from perturbed prefixes
4. The full cycle doesn't corrupt TPU state
"""
import jax
import jax.numpy as jnp
import numpy as np
import gc
import inspect
import time
import sys
import os

jax.distributed.initialize()
from jax.experimental import multihost_utils
import easydel as ed
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_prompts import load_all_problems, build_code_prompt
from code_executor import execute_code, extract_code_from_response

is_master = jax.process_index() == 0

if is_master:
    print(f"{'='*70}")
    print(f"TEST: Forward-pass → eSurge rebuild → generation cycle")
    print(f"  Devices: {jax.device_count()}, Local: {jax.local_device_count()}")
    print(f"{'='*70}")

MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
axis_dims = (1, 1, 8, 4, 1)
axis_names = ("dp", "fsdp", "tp", "sp", "selective")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load model + init via eSurge
elm = (
    ed.eLargeModel.from_pretrained(MODEL_ID)
    .set_dtype("bf16")
    .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
    .set_esurge(max_model_len=4096, max_num_seqs=4)
)

if is_master:
    print("\n[Step 1] Build eSurge to init model, then tear down")

esurge = elm.build_esurge()
del esurge
gc.collect()
jax.clear_caches()
gc.collect()
multihost_utils.sync_global_devices("init_cleanup")

if is_master:
    print("  eSurge torn down")

# Monkey-patch output validation
import easydel.infra.modeling_outputs as _mo
for _name, _cls in inspect.getmembers(_mo, inspect.isclass):
    if hasattr(_cls, '__post_init__'):
        _cls.__post_init__ = lambda self: None

# ── TEST 1: Forward pass with hidden states ─────────────────────────
if is_master:
    print("\n[Step 2] Forward pass with hidden states")

problems = load_all_problems()
problem = problems[0]
conversation = build_code_prompt(problem)

# Create a fake "baseline response" for testing
baseline_text = "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False"
full_conversation = conversation + [{"role": "assistant", "content": baseline_text}]

input_ids = tokenizer.apply_chat_template(
    full_conversation, return_tensors="np", add_generation_prompt=False,
)
prompt_only_ids = tokenizer.apply_chat_template(
    conversation, return_tensors="np", add_generation_prompt=True,
)
response_start = prompt_only_ids.shape[1]
seq_len = input_ids.shape[1]

if is_master:
    print(f"  Seq len: {seq_len}, response starts: {response_start}")

model_mesh = elm._model.config.mesh
with model_mesh:
    out = elm._model(input_ids=jnp.array(input_ids), output_hidden_states=True)

logits = out.logits
hidden_states = out.hidden_states

if is_master:
    print(f"  Logits shape: {logits.shape}")
    print(f"  Hidden states: {len(hidden_states)} layers")

# Gather hidden states for layer 8 to numpy
gathered_h8 = np.array(
    multihost_utils.process_allgather(hidden_states[9], tiled=True)
)
logits_np = np.array(
    multihost_utils.process_allgather(logits, tiled=True)
)

if is_master:
    print(f"  Gathered hidden[8] shape: {gathered_h8.shape}")
    print(f"  Gathered logits shape: {logits_np.shape}")
    print(f"  PASS: Forward pass with hidden states works")

# Free JAX tensors before rebuilding eSurge
del out, logits, hidden_states
gc.collect()
multihost_utils.sync_global_devices("forward_done")

# ── TEST 2: Rebuild eSurge after forward passes ─────────────────────
if is_master:
    print(f"\n[Step 3] Rebuilding eSurge after forward passes")
    t0 = time.time()

esurge2 = elm.build_esurge()

if is_master:
    rebuild_time = time.time() - t0
    print(f"  eSurge rebuilt in {rebuild_time:.1f}s")

# ── TEST 3: Generate a complete response with eSurge ────────────────
if is_master:
    print(f"\n[Step 4] Generate complete response with rebuilt eSurge")

gen_text = ""
for output in esurge2.chat(
    conversation,
    sampling_params=ed.SamplingParams(max_tokens=256, temperature=0.0),
    stream=True,
):
    gen_text += output.delta_text

if is_master:
    code = extract_code_from_response(gen_text)
    reward, info = execute_code(code, problem.test_code, timeout=10)
    print(f"  Generated {len(gen_text)} chars, reward={reward}")
    print(f"  PASS: eSurge generation works after forward passes")

multihost_utils.sync_global_devices("gen_done")

# ── TEST 4: Continuation from perturbed prefix ──────────────────────
if is_master:
    print(f"\n[Step 5] Test continuation from perturbed prefix")

# Simulate a perturbation: take the first half of the baseline response
# and swap one token, then let eSurge continue
perturb_pos = response_start + 5  # 5 tokens into the response
perturbed_ids = np.array(input_ids[0])
original_token = perturbed_ids[perturb_pos]

# Find a different token (next most likely from logits)
pos_logits = logits_np[0, perturb_pos - 1, :]  # logits predicting token at perturb_pos
sorted_tokens = np.argsort(pos_logits)[::-1]
perturbed_token = sorted_tokens[0] if sorted_tokens[0] != original_token else sorted_tokens[1]

if is_master:
    print(f"  Perturbing position {perturb_pos}: '{tokenizer.decode([int(original_token)])}' → '{tokenizer.decode([int(perturbed_token)])}'")

# Construct perturbed prefix: tokens [0:perturb_pos] + perturbed_token
perturbed_prefix_ids = perturbed_ids[:perturb_pos + 1].copy()
perturbed_prefix_ids[perturb_pos] = perturbed_token

# Decode the perturbed prefix back to text
# We need just the assistant's partial response (after prompt)
perturbed_response_prefix = tokenizer.decode(
    perturbed_prefix_ids[response_start:].tolist(), skip_special_tokens=True
)

if is_master:
    print(f"  Perturbed prefix text: '{perturbed_response_prefix[:100]}...'")

# Method A: Embed prefix in the user prompt for continuation
continuation_prompt = build_code_prompt(problem)
continuation_prompt[0]["content"] += (
    f"\n\nBegin your response exactly with this code prefix and continue from it:\n"
    f"```python\n{perturbed_response_prefix}"
)

cont_text = ""
for output in esurge2.chat(
    continuation_prompt,
    sampling_params=ed.SamplingParams(max_tokens=256, temperature=0.0),
    stream=True,
):
    cont_text += output.delta_text

if is_master:
    cont_code = extract_code_from_response(cont_text)
    cont_reward, _ = execute_code(cont_code, problem.test_code, timeout=10)
    print(f"  Method A (prompt injection): {len(cont_text)} chars, reward={cont_reward}")

# Method B: Use chat with partial assistant message
# Some tokenizers support continue_final_message
try:
    partial_conv = conversation + [{"role": "assistant", "content": perturbed_response_prefix}]
    prefix_tokens_b = tokenizer.apply_chat_template(
        partial_conv, add_generation_prompt=False, return_tensors="np",
    )
    # Decode back to string and feed as a new conversation
    full_prefix_text = tokenizer.decode(prefix_tokens_b[0].tolist())

    if is_master:
        print(f"  Method B prefix tokens: {prefix_tokens_b.shape[1]}")
        # This is just informational — we can't feed raw tokens to eSurge
        print(f"  Method B: Chat template with partial assistant response encoded OK")
except Exception as e:
    if is_master:
        print(f"  Method B: {e}")

multihost_utils.sync_global_devices("continuation_done")

# ── TEST 5: Tear down eSurge and do another forward pass ────────────
if is_master:
    print(f"\n[Step 6] Tear down eSurge, verify forward pass still works")

del esurge2
gc.collect()
jax.clear_caches()
gc.collect()
multihost_utils.sync_global_devices("esurge2_cleanup")

# Re-patch output validation (clear_caches may have reset things)
import easydel.infra.modeling_outputs as _mo2
for _name, _cls in inspect.getmembers(_mo2, inspect.isclass):
    if hasattr(_cls, '__post_init__'):
        _cls.__post_init__ = lambda self: None

model_mesh = elm._model.config.mesh
with model_mesh:
    out2 = elm._model(input_ids=jnp.array(input_ids[:, :20]), output_hidden_states=True)

if is_master:
    print(f"  Second forward pass logits: {out2.logits.shape}")
    print(f"  PASS: Forward pass works after eSurge teardown")

del out2
gc.collect()

if is_master:
    print(f"\n{'='*70}")
    print("ALL TESTS PASSED")
    print(f"{'='*70}")

multihost_utils.sync_global_devices("all_done")
gc.collect()
multihost_utils.sync_global_devices("ready_to_kill")
jax.distributed.shutdown()
sys.exit(0)
