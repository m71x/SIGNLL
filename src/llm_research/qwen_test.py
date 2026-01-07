# inference_from_sharded.py
# Run this on ALL workers of your TPU pod

import os
import json
import numpy as np

CACHE_DIR = "/dev/shm/huggingface"
CHECKPOINT_DIR = "/path/to/sharded_checkpoint"  # Same as OUTPUT_DIR above

os.environ["HF_HOME"] = CACHE_DIR
os.environ["PJRT_DEVICE"] = "TPU"

# Initialize distributed FIRST
import jax
jax.distributed.initialize()

print(f"JAX distributed initialized:")
print(f"  Process index: {jax.process_index()}")
print(f"  Process count: {jax.process_count()}")
print(f"  Local devices: {jax.local_device_count()}")
print(f"  Global devices: {jax.device_count()}")

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from transformers import AutoTokenizer, AutoConfig
from flax import linen as nn
import time

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
FSDP_SIZE = 8
TP_SIZE = 4
SP_SIZE = 1
MAX_NEW_TOKENS = 30

# ----------------------------------------------------------------------
# LOAD TOKENIZER
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
else:
    time.sleep(5)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------------
# CREATE GLOBAL MESH
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("DEVICE AND MESH SETUP")
    print("="*80)

all_devices = jax.devices()
device_array = mesh_utils.create_device_mesh((FSDP_SIZE, TP_SIZE, SP_SIZE), devices=all_devices)
mesh = Mesh(device_array, axis_names=('fsdp', 'tp', 'sp'))

if jax.process_index() == 0:
    print(f"Mesh: {mesh.devices.shape} = {mesh.devices.size} devices")

# ----------------------------------------------------------------------
# LOAD SHARDED PARAMETERS
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("LOADING PRE-SHARDED PARAMETERS")
    print("="*80)

# Load index
with open(os.path.join(CHECKPOINT_DIR, "params_index.json"), "r") as f:
    params_index = json.load(f)

def spec_from_tuple(spec_tuple):
    """Convert tuple like ('fsdp', 'tp') to PartitionSpec."""
    return P(*spec_tuple)

params = {}
with mesh:
    for name, info in params_index.items():
        filepath = os.path.join(CHECKPOINT_DIR, info["file"])
        
        # Only coordinator loads from disk
        if jax.process_index() == 0:
            arr = np.load(filepath)
            arr = jnp.array(arr, dtype=jnp.bfloat16)
        else:
            # Other processes create placeholder
            arr = jnp.zeros(info["shape"], dtype=jnp.bfloat16)
        
        # Create sharding and distribute
        spec = spec_from_tuple(info["sharding"])
        sharding = NamedSharding(mesh, spec)
        
        # Shard across all devices
        arr = jax.device_put(arr, sharding)
        
        # Store in nested dict structure
        keys = name.split(".")
        current = params
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = arr
        
        if jax.process_index() == 0:
            print(f"  Loaded: {name} -> {arr.shape}, spec={spec}")

if jax.process_index() == 0:
    print("✓ All parameters loaded and sharded!")

# ----------------------------------------------------------------------
# PARAMETER SHARDING ANALYSIS
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("PARAMETER SHARDING ANALYSIS")
    print("="*80)
    
    # Check a sample weight
    for name, info in list(params_index.items())[:3]:
        keys = name.split(".")
        val = params
        for k in keys:
            val = val[k]
        print(f"  {name}: shape={val.shape}, sharding={type(val.sharding).__name__}")
        if hasattr(val.sharding, 'spec'):
            print(f"    Spec: {val.sharding.spec}")

# ----------------------------------------------------------------------
# LOAD MODEL CLASS (for inference)
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("SETTING UP INFERENCE")
    print("="*80)

# Now use EasyDeL or custom Flax model for forward pass
from easydel import AutoEasyDeLModelForCausalLM, PartitionAxis

config = AutoConfig.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)

# Create model structure without loading weights
partition_axis = PartitionAxis(
    fully_sharded_data_parallel_axis="fsdp",
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

# Load just the model class (weights already in params)
with mesh:
    model, _ = AutoEasyDeLModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
        sharding_axis_names=('fsdp', 'tp', 'sp'),
        partition_axis=partition_axis,
        trust_remote_code=True,
    )

# Use our pre-loaded params instead
# (This bypasses EasyDeL's loading which was causing OOM)

if jax.process_index() == 0:
    print("✓ Model ready for inference!")

# ----------------------------------------------------------------------
# GENERATION
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(formatted_prompt, return_tensors="np", padding=True)
input_ids = jnp.array(inputs["input_ids"])

with mesh:
    input_sharding = NamedSharding(mesh, P(None, None))
    input_ids = jax.device_put(input_ids, input_sharding)
    current_ids = input_ids
    
    for step in range(MAX_NEW_TOKENS):
        outputs = model(current_ids, params=params)
        next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1, keepdims=True)
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        if next_token[0, 0] == tokenizer.eos_token_id:
            break

output_ids_np = jax.device_get(current_ids)
generated_text = tokenizer.decode(output_ids_np[0], skip_special_tokens=True)

if jax.process_index() == 0:
    print("\n" + "="*80)
    print("GENERATED OUTPUT")
    print("="*80)
    print(generated_text)
    print("="*80)