import jax
import jax.numpy as jnp
import numpy as np
import sys
import gc
import inspect

# 1. Initialize distributed system
jax.distributed.initialize()
from jax.experimental import multihost_utils
import easydel as ed
from transformers import AutoTokenizer

is_master = jax.process_index() == 0

if is_master:
    print("Starting model initialization...")

# 2. Load tokenizer and model
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

axis_dims = (1, 1, 8, 4, 1)
axis_names = ("dp", "fsdp", "tp", "sp", "selective")

elm = (
    ed.eLargeModel.from_pretrained(model_id)
    .set_dtype("bf16")
    .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
    .set_esurge(max_model_len=4096, max_num_seqs=32)
)

esurge = elm.build_esurge()

# ── STEP 1: GENERATION ─────────────────────────────────────────────────
prompt = "Explain the difference between TCP and UDP in one paragraph."
conversation = [{"role": "user", "content": prompt}]

if is_master:
    print(f"\nPROMPT: {prompt}\nResponse: ", end="", flush=True)

# IMPORTANT: Accumulate generated_text on ALL workers, not just master.
# All workers must have identical text for the forward pass, otherwise
# tokenization produces different sequences on each worker → TPU halt.
generated_text = ""
for output in esurge.chat(
    conversation,
    sampling_params=ed.SamplingParams(max_tokens=256),
    stream=True,
):
    generated_text += output.delta_text
    if is_master:
        print(output.delta_text, end="", flush=True)

if is_master:
    print("\n")

# ── STEP 2: ACTIVATION EXTRACTION (Forward Pass) ──────────────────────
if is_master:
    print("--- Shutting down eSurge before forward pass ---")

# Stop the eSurge engine to free up TPU resources
del esurge
gc.collect()
multihost_utils.sync_global_devices("esurge_stopped")

if is_master:
    print("--- Feeding response back into the model for activation extraction ---")

full_conversation = conversation + [{"role": "assistant", "content": generated_text}]

input_ids = tokenizer.apply_chat_template(
    full_conversation,
    return_tensors="np",
    add_generation_prompt=False,
)

if is_master:
    print(f"Tokenized sequence length: {input_ids.shape[1]}")

# Monkey-patch EasyDeL output dataclass validation for multi-host.
# In eager mode on multi-host, intermediate arrays are sharded across hosts
# and fail is_fully_replicated/is_fully_addressable checks in __post_init__.
# These are just validation asserts — not needed for the computation itself.
import easydel.infra.modeling_outputs as _mo
for _name, _cls in inspect.getmembers(_mo, inspect.isclass):
    if hasattr(_cls, '__post_init__'):
        _cls.__post_init__ = lambda self: None

if is_master:
    print("Patched output validation for multi-host forward pass")

# Use the mesh from the model's own config (guarantees matching device ordering)
model_mesh = elm._model.config.mesh

# Forward pass in eager mode inside the mesh context
with model_mesh:
    if is_master:
        print("Running forward pass...")
    model_outputs = elm._model(
        input_ids=jnp.array(input_ids),
        output_hidden_states=True,
    )

# Last layer activations  →  shape: (batch, seq_len, hidden_dim)
last_layer_activations = model_outputs.hidden_states[-1]
logits = model_outputs.logits

# ALL workers must participate in sharded array ops (float, mean, softmax, etc.)
# Only print() calls are guarded by is_master.
num_layers = len(model_outputs.hidden_states)
act = last_layer_activations[0]  # (seq_len, hidden_dim)
mean_val = float(jnp.mean(act))
std_val = float(jnp.std(act))
min_val = float(jnp.min(act))
max_val = float(jnp.max(act))

probs = jax.nn.softmax(logits, axis=-1)
seq_len = input_ids.shape[1]
last_n = min(5, seq_len)

# Pre-compute per-token data on ALL workers
token_data = []
for pos in range(seq_len - last_n, seq_len):
    token_id = int(input_ids[0, pos])
    top_k_indices = jnp.argsort(probs[0, pos, :])[-3:][::-1]
    top_k_probs = probs[0, pos, top_k_indices]
    act_slice = last_layer_activations[0, pos, :5]
    # Materialize values on all workers
    top_k_ids = [int(top_k_indices[k]) for k in range(3)]
    top_k_ps = [float(top_k_probs[k]) for k in range(3)]
    act_vals = act_slice.tolist()
    token_data.append((pos, token_id, top_k_ids, top_k_ps, act_vals))

# Only master prints results
if is_master:
    print(f"Number of hidden-state layers returned: {num_layers}")
    print(f"Last-layer activation shape: {last_layer_activations.shape}")

    print(f"\nLast-layer activation statistics:")
    print(f"  mean : {mean_val:.6f}")
    print(f"  std  : {std_val:.6f}")
    print(f"  min  : {min_val:.6f}")
    print(f"  max  : {max_val:.6f}")

    print(f"\nPer-token detail (last {last_n} positions):")
    for pos, token_id, top_k_ids, top_k_ps, act_vals in token_data:
        token_str = tokenizer.decode([token_id])
        print(f"\n  Position {pos} — token: '{token_str}'")
        print(f"    Activation (first 5 dims): {act_vals}")
        for k in range(3):
            t_name = tokenizer.decode([top_k_ids[k]])
            print(f"    Top-{k+1} prediction: '{t_name}' ({top_k_ps[k]*100:.2f}%)")

# ── CLEANUP ────────────────────────────────────────────────────────────
if is_master:
    print("\nCleaning up...")
del elm
gc.collect()
multihost_utils.sync_global_devices("ready_to_kill")
jax.distributed.shutdown()
sys.exit(0)