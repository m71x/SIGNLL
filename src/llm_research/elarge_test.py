import jax
import jax.numpy as jnp
import numpy as np
import sys
import gc
from jax.sharding import Mesh

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

# Create the JAX mesh (needed for direct forward passes on the sharded model)
devices = np.array(jax.devices()).reshape(axis_dims)
mesh = Mesh(devices, axis_names)

# ── STEP 1: GENERATION ─────────────────────────────────────────────────
prompt = "Explain the difference between TCP and UDP in one paragraph."
conversation = [{"role": "user", "content": prompt}]

if is_master:
    print(f"\nPROMPT: {prompt}\nResponse: ", end="", flush=True)

generated_text = ""
for output in esurge.chat(
    conversation,
    sampling_params=ed.SamplingParams(max_tokens=256),
    stream=True,
):
    if is_master:
        print(output.delta_text, end="", flush=True)
        generated_text += output.delta_text

if is_master:
    print("\n")

# ── STEP 2: ACTIVATION EXTRACTION (Forward Pass) ──────────────────────
# Build the full conversation (user prompt + model response) and tokenize
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

# Forward pass with hidden-state capture (must run inside mesh context)
with mesh:
    model_outputs = elm._model(
        input_ids=jnp.array(input_ids),
        output_hidden_states=True,
    )

# Last layer activations  →  shape: (batch, seq_len, hidden_dim)
last_layer_activations = model_outputs.hidden_states[-1]

if is_master:
    print(f"Number of hidden-state layers returned: {len(model_outputs.hidden_states)}")
    print(f"Last-layer activation shape: {last_layer_activations.shape}")

    # ── Summary statistics ──
    act = last_layer_activations[0]  # (seq_len, hidden_dim)
    print(f"\nLast-layer activation statistics:")
    print(f"  mean : {float(jnp.mean(act)):.6f}")
    print(f"  std  : {float(jnp.std(act)):.6f}")
    print(f"  min  : {float(jnp.min(act)):.6f}")
    print(f"  max  : {float(jnp.max(act)):.6f}")

    # ── Per-token detail for the last 5 tokens ──
    probs = jax.nn.softmax(model_outputs.logits, axis=-1)
    seq_len = input_ids.shape[1]
    last_n = min(5, seq_len)

    print(f"\nPer-token detail (last {last_n} positions):")
    for pos in range(seq_len - last_n, seq_len):
        token_id = int(input_ids[0, pos])
        token_str = tokenizer.decode([token_id])

        top_k_indices = jnp.argsort(probs[0, pos, :])[-3:][::-1]
        top_k_probs = probs[0, pos, top_k_indices]

        print(f"\n  Position {pos} — token: '{token_str}'")
        print(f"    Activation (first 5 dims): {last_layer_activations[0, pos, :5]}")
        for k in range(3):
            t_name = tokenizer.decode([int(top_k_indices[k])])
            print(f"    Top-{k+1} prediction: '{t_name}' ({float(top_k_probs[k])*100:.2f}%)")

# ── CLEANUP ────────────────────────────────────────────────────────────
if is_master:
    print("\nCleaning up...")
del esurge, elm
gc.collect()
multihost_utils.sync_global_devices("ready_to_kill")
jax.distributed.shutdown()
sys.exit(0)