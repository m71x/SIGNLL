import os
import re

# Ensure TPU is used
os.environ["PJRT_DEVICE"] = "TPU"

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoConfig, FlaxLlamaForCausalLM
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax.traverse_util import flatten_dict, unflatten_dict

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# TPU MESH CONFIGURATION (Targeting 32 Chips / 64 TensorCores)
# We use 8-way Tensor Parallelism (TP) to shard the 64GB weights.
# 4-way Data Parallelism (DP) replicates that across the remaining chips.
TP_SIZE = 8
DP_SIZE = 4 

# ----------------------------------------------------------------------
# 2. SETUP DEVICE MESH
# ----------------------------------------------------------------------
devices = jax.devices()
if len(devices) != TP_SIZE * DP_SIZE:
    print(f"Warning: Found {len(devices)} devices, but configured for {TP_SIZE * DP_SIZE}.")

# Mapping DP and TP to physical chips
device_mesh = mesh_utils.create_device_mesh((DP_SIZE, TP_SIZE))
mesh = Mesh(device_mesh, axis_names=('data', 'model'))

print(f"Constructed Mesh for 32 Chips: {mesh}")

# ----------------------------------------------------------------------
# 3. LOAD MODEL & TOKENIZER (WITH WEIGHT CONVERSION)
# ----------------------------------------------------------------------
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading Configuration (Spoofing Llama)...")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.model_type = "llama" # Architecturally compatible

print("Loading and Converting Weights (PyTorch -> Flax)...")
# FIX: Added from_pt=True to handle the absence of flax_model.msgpack
model = FlaxLlamaForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    dtype=jnp.bfloat16,
    from_pt=True,           # <--- CRITICAL FIX
    trust_remote_code=True,
    _do_init=False          # Skip random init to save memory
)

# ----------------------------------------------------------------------
# 4. SHARDING RULES
# ----------------------------------------------------------------------
def get_sharding_rules(params):
    flat_params = flatten_dict(params)
    flat_sharding = {}
    
    for key, value in flat_params.items():
        name = key[-1]
        path = "/".join(key)
        
        # Default: Replicate
        spec = P(None)
        
        # Vocab sharding
        if "embed_tokens" in path:
            spec = P(None, "model")
        # Attention sharding (Heads)
        elif "self_attn" in path and "kernel" in name:
            if any(x in path for x in ["q_proj", "k_proj", "v_proj"]):
                spec = P(None, "model")
            elif "o_proj" in path:
                spec = P("model", None)
        # MLP sharding
        elif "mlp" in path and "kernel" in name:
            if "gate_proj" in path or "up_proj" in path:
                spec = P(None, "model")
            elif "down_proj" in path:
                spec = P("model", None)
        # Output sharding
        elif "lm_head" in path and "kernel" in name:
            spec = P("model", None)

        flat_sharding[key] = NamedSharding(mesh, spec)
        
    return unflatten_dict(flat_sharding)

print("Distributing 32B parameters across 32 TPU chips...")
sharding_specs = get_sharding_rules(model.params)
model_params = jax.device_put(model.params, sharding_specs)
model.params = None # Clear CPU shadow copy

# ----------------------------------------------------------------------
# 5. INFERENCE
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."
inputs = tokenizer(prompt, return_tensors="np")

# Shard input across Data Parallel axis
input_sharding = NamedSharding(mesh, P('data', None))
input_ids = jax.device_put(jnp.array(inputs["input_ids"]), input_sharding)

@jax.jit
def prefill(params, input_ids):
    outputs = model(input_ids=input_ids, params=params, use_cache=True)
    next_token = jnp.argmax(outputs.logits[:, -1], axis=-1)
    return outputs.past_key_values, next_token

@jax.jit
def decode_step(carry, _):
    params, past_kv, token = carry
    outputs = model(input_ids=token[:, None], params=params, past_key_values=past_kv, use_cache=True)
    next_token = jnp.argmax(outputs.logits[:, -1], axis=-1)
    return (params, outputs.past_key_values, next_token), next_token

print("Running Inference...")
past_kv, next_token = prefill(model_params, input_ids)

_, generated = jax.lax.scan(
    decode_step,
    (model_params, past_kv, next_token),
    xs=None,
    length=MAX_NEW_TOKENS - 1
)

# ----------------------------------------------------------------------
# 6. OUTPUT
# ----------------------------------------------------------------------
all_tokens = jnp.concatenate([input_ids, next_token[:, None], generated.T], axis=1)
print("-" * 40)
print(tokenizer.decode(np.array(all_tokens[0]), skip_special_tokens=True))
print("-" * 40)