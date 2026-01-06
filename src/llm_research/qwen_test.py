# ============================================================
# Qwen-2.5-Coder-32B EasyDeL TPU Script (v0.2.0.2 COMPATIBLE)
# ============================================================

import os
os.environ["PJRT_DEVICE"] = "TPU"

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from transformers import AutoTokenizer

from easydel import AutoEasyDelModelForCausalLM

# ============================================================
# 1. CONFIG
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
DTYPE = jnp.bfloat16

TP_SIZE = 8
DP_SIZE = 4
TOTAL_DEVICES = TP_SIZE * DP_SIZE

MAX_SEQ_LEN = 4096

# ============================================================
# 2. TPU MESH
# ============================================================

devices = jax.devices()
assert len(devices) == TOTAL_DEVICES, (
    f"Expected {TOTAL_DEVICES} devices, got {len(devices)}"
)

mesh = Mesh(
    mesh_utils.create_device_mesh((DP_SIZE, TP_SIZE)),
    axis_names=("data", "model")
)

print("✅ TPU mesh created")

# ============================================================
# 3. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 4. LOAD MODEL (v0.2 API)
# ============================================================

with mesh:
    model = AutoEasyDelModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        shard_parameters=True,
        trust_remote_code=True,
        max_position_embeddings=MAX_SEQ_LEN,
    )

print("✅ Qwen-2.5-Coder-32B loaded & sharded")

# ============================================================
# 5. FORWARD WITH LOGIT ACCESS
# ============================================================

@jax.jit
def forward(params, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        params=params,
        deterministic=True,
        output_hidden_states=True,
    )
    return outputs.logits, outputs.hidden_states

# ============================================================
# 6. INFERENCE TEST
# ============================================================

prompt = "Write a high-performance C++ thread pool."

inputs = tokenizer(
    prompt,
    return_tensors="np",
    padding="max_length",
    max_length=512,
    truncation=True,
)

logits, hidden = forward(
    model.params,
    jnp.array(inputs["input_ids"]),
    jnp.array(inputs["attention_mask"]),
)

next_token = jnp.argmax(logits[:, -1], axis=-1)
print("\n--- Inference Test ---")
print(tokenizer.decode(next_token.tolist()))
print("----------------------")

# ============================================================
# 7. LOGIT MODIFICATION (RESEARCH)
# ============================================================

def apply_logit_penalty(logits):
    return logits + jnp.where(logits < -10.0, -5.0, 0.0)

@jax.jit
def forward_with_custom_logits(params, input_ids, attention_mask):
    logits, hidden_states = forward(
        params, input_ids, attention_mask
    )
    return apply_logit_penalty(logits), hidden_states

print("✅ Ready for fine-tuning & research (v0.2.0.2)")
