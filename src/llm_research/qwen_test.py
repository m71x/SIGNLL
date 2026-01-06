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

# TPU MESH CONFIGURATION (Targeting 32 Chips)
# We use 8-way Tensor Parallelism (TP) to fit the huge weights
# and 4-way Data Parallelism (DP) to utilize the remaining chips.
TP_SIZE = 8
DP_SIZE = 4 

# ----------------------------------------------------------------------
# 2. SETUP DEVICE MESH
# ----------------------------------------------------------------------
# Create the mesh of devices (32 chips)
devices = jax.devices()
if len(devices) != TP_SIZE * DP_SIZE:
    print(f"Warning: You have {len(devices)} devices, but configured mesh for {TP_SIZE*DP_SIZE}.")
    # Fallback to creating a mesh that fits available devices if possible
    
device_mesh = mesh_utils.create_device_mesh((DP_SIZE, TP_SIZE))
mesh = Mesh(device_mesh, axis_names=('data', 'model'))

print(f"Constructed Mesh: {mesh}")

# ----------------------------------------------------------------------
# 3. LOAD MODEL & TOKENIZER (CPU FIRST)
# ----------------------------------------------------------------------
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading Configuration (Spoofing Llama)...")
# FIX: Qwen2 is architecturally Llama. We swap the config type so Flax loads it.
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.model_type = "llama"

print("Loading Model to Host (CPU)...")
# We load to CPU first to avoid OOM on a single TPU chip before sharding
model = FlaxLlamaForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    dtype=jnp.bfloat16,
    from_pt=True,
    trust_remote_code=True,
    _do_init=False 
)

# ----------------------------------------------------------------------
# 4. SHARDING RULES (CRITICAL FOR 32B)
# ----------------------------------------------------------------------
# This function maps parameter names to PartitionSpecs for 8-way Model Parallelism
def get_sharding_rules(params):
    flat_params = flatten_dict(params)
    flat_sharding = {}
    
    for key, value in flat_params.items():
        # Key is a tuple like ('model', 'layers', '0', 'self_attn', 'q_proj', 'kernel')
        name = key[-1]
        path = "/".join(key)
        
        # Default: Replicate (None)
        spec = P(None)
        
        # Rule 1: Embeddings (Vocab Parallel)
        if "embed_tokens" in path:
            spec = P(None, "model")
            
        # Rule 2: Attention Weights (Heads Parallel)
        elif "self_attn" in path and "kernel" in name:
            # Q, K, V projections: Split headers
            if any(x in path for x in ["q_proj", "k_proj", "v_proj"]):
                spec = P(None, "model")
            # Output projection: Split input dimension
            elif "o_proj" in path:
                spec = P("model", None)
                
        # Rule 3: MLP (Tensor Parallel)
        elif "mlp" in path and "kernel" in name:
            if "gate_proj" in path or "up_proj" in path:
                spec = P(None, "model")
            elif "down_proj" in path:
                spec = P("model", None)
                
        # Rule 4: LM Head
        elif "lm_head" in path and "kernel" in name:
            spec = P("model", None)

        flat_sharding[key] = NamedSharding(mesh, spec)
        
    return unflatten_dict(flat_sharding)

print("Sharding Model Parameters to TPUs...")
sharding_specs = get_sharding_rules(model.params)

# Move parameters from CPU to TPU with the defined sharding layout
model_params = jax.device_put(model.params, sharding_specs)

# Free CPU memory
model.params = None 

# ----------------------------------------------------------------------
# 5. INPUT PROCESSING
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."
inputs = tokenizer(prompt, return_tensors="np")
input_ids_np = inputs["input_ids"]

# Replicate input_ids across the 'data' axis
input_sharding = NamedSharding(mesh, P('data', None))
input_ids = jax.device_put(jnp.array(input_ids_np), input_sharding)

# ----------------------------------------------------------------------
# 6. INFERENCE FUNCTIONS (JIT COMPILED)
# ----------------------------------------------------------------------

# We bind the model params specifically to ensure JIT knows the layout
def forward_fn(params, input_ids, past_key_values=None):
    return model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        params=params,
        use_cache=True,
    )

@jax.jit
def prefill(params, input_ids):
    outputs = forward_fn(params, input_ids)
    next_token = jnp.argmax(outputs.logits[:, -1], axis=-1)
    return outputs.past_key_values, next_token

@jax.jit
def decode_step(carry, _):
    params, past_kv, token = carry
    
    outputs = forward_fn(params, token[:, None], past_kv)
    next_token = jnp.argmax(outputs.logits[:, -1], axis=-1)
    
    return (params, outputs.past_key_values, next_token), next_token

# ----------------------------------------------------------------------
# 7. EXECUTION
# ----------------------------------------------------------------------
print("Running Prefill...")
past_kv, next_token = prefill(model_params, input_ids)

print("Running Decode Loop...")
# Scan requires carrying the params through if we want them available inside
(final_params, final_kv, last_token), generated = jax.lax.scan(
    decode_step,
    (model_params, past_kv, next_token),
    xs=None,
    length=MAX_NEW_TOKENS
)

# ----------------------------------------------------------------------
# 8. OUTPUT
# ----------------------------------------------------------------------
# Gather results back to CPU for decoding
generated_cpu = np.array(generated)
next_token_cpu = np.array(next_token)
input_ids_cpu = np.array(input_ids)

all_tokens = np.concatenate(
    [input_ids_cpu, next_token_cpu[:, None], generated_cpu.T],
    axis=1,
)

print("-" * 40)
print(tokenizer.decode(all_tokens[0], skip_special_tokens=True))
print("-" * 40)