# ============================================================================
# MUST COME FIRST — ENVIRONMENT SETUP
# ============================================================================
import os

CACHE_DIR = "/dev/shm/huggingface"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"

os.environ["TMPDIR"] = CACHE_DIR
os.environ["TEMP"] = CACHE_DIR
os.environ["TMP"] = CACHE_DIR

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PJRT_DEVICE"] = "TPU"

# ============================================================================
# INITIALIZE JAX DISTRIBUTED BEFORE ANY JAX/EASYDEL IMPORTS
# ============================================================================
import jax

jax.distributed.initialize()

print(f"JAX distributed initialized:")
print(f"  Process index: {jax.process_index()}")
print(f"  Process count: {jax.process_count()}")
print(f"  Local devices: {jax.local_device_count()}")
print(f"  Global devices: {jax.device_count()}")

# ============================================================================
# IMPORTS (AFTER DISTRIBUTED INIT)
# ============================================================================
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import time

from easydel import (
    AutoEasyDeLModelForCausalLM,
    PartitionAxis,
)
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# TPU MESH CONFIGURATION (32 Chips = 8 FSDP x 4 TP x 1 SP)
FSDP_SIZE = 8
TP_SIZE = 4
SP_SIZE = 1

# ----------------------------------------------------------------------
# 2. LOAD TOKENIZER (Coordinator first, then others)
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
else:
    time.sleep(10)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------------
# 3. CREATE GLOBAL MESH (ACROSS ALL HOSTS)
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("DEVICE AND MESH SETUP")
    print("="*80)

# Use ALL devices across ALL hosts - this is critical for multi-host
all_devices = jax.devices()
total_devices = len(all_devices)
expected_devices = FSDP_SIZE * TP_SIZE * SP_SIZE

if jax.process_index() == 0:
    print(f"Total global devices: {total_devices}")
    print(f"Expected devices: {expected_devices}")

if total_devices != expected_devices:
    raise RuntimeError(
        f"Device count mismatch: got {total_devices}, expected {expected_devices}"
    )

# Create a GLOBAL mesh spanning all hosts
device_array = mesh_utils.create_device_mesh(
    (FSDP_SIZE, TP_SIZE, SP_SIZE),
    devices=all_devices,
)

mesh = Mesh(device_array, axis_names=('fsdp', 'tp', 'sp'))

if jax.process_index() == 0:
    print(f"Mesh shape: {mesh.devices.shape}")
    print(f"Mesh axis names: {mesh.axis_names}")
    print(f"Total devices in mesh: {mesh.devices.size}")

# Partition axis configuration
partition_axis = PartitionAxis(
    fully_sharded_data_parallel_axis="fsdp",
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

# ----------------------------------------------------------------------
# 4. PARTITION RULES (Correct format using PartitionSpec)
# ----------------------------------------------------------------------
# EasyDeL expects tuples of (pattern_string, PartitionSpec)
# NOT compiled regex or custom dataclasses
partition_rules = (
    ("embed_tokens", P("fsdp", "tp")),
    ("q_proj", P("fsdp", "tp")),
    ("k_proj", P("fsdp", "tp")),
    ("v_proj", P("fsdp", "tp")),
    ("o_proj", P("tp", "fsdp")),
    ("gate_proj", P("fsdp", "tp")),
    ("up_proj", P("fsdp", "tp")),
    ("down_proj", P("tp", "fsdp")),
    ("lm_head", P("fsdp", "tp")),
    ("input_layernorm", P(None,)),
    ("post_attention_layernorm", P(None,)),
    ("norm", P(None,)),
    (".*", P("fsdp", "tp")),  # Fallback for any unmatched
)

if jax.process_index() == 0:
    print(f"\n✅ Partition Rules Configured (using PartitionSpec)")

# ----------------------------------------------------------------------
# 5. LOAD MODEL WITH PROPER SHARDING
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("LOADING MODEL WITH SHARDING")
    print("="*80)

with mesh:
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        # Sharding configuration
        sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
        sharding_axis_names=('fsdp', 'tp', 'sp'),
        partition_axis=partition_axis,
        partition_rules=partition_rules,
        # Memory optimization
        shard_attention_computation=True,
        input_shape=(1, 1),  # Minimal during loading
        # Model config
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        config_kwargs={
            "gradient_checkpointing": "",
            "use_scan_mlp": False,
        }
    )

if jax.process_index() == 0:
    print("✓ Model loaded successfully across all hosts!")

# ----------------------------------------------------------------------
# 6. VERIFY SHARDING (Debug)
# ----------------------------------------------------------------------
if jax.process_index() == 0 and params is not None:
    print("\n" + "="*80)
    print("PARAMETER SHARDING VERIFICATION")
    print("="*80)

    import jax.tree_util as tree_util
    leaves_with_paths, _ = tree_util.tree_flatten_with_path(params)

    print(f"\n🔍 Checking sample weights:")
    checked = 0
    for path, val in leaves_with_paths:
        path_str = tree_util.keystr(path)
        if checked < 5 and "kernel" in path_str:
            print(f"\n  Weight: {path_str}")
            print(f"    Global shape: {val.shape}")
            sharding_type = type(val.sharding).__name__
            print(f"    Sharding: {sharding_type}")
            if hasattr(val.sharding, 'spec'):
                print(f"    Spec: {val.sharding.spec}")
            checked += 1

# ----------------------------------------------------------------------
# 7. PREPARE INPUT WITH PROPER SHARDING
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."

if hasattr(tokenizer, "apply_chat_template"):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
else:
    formatted_prompt = prompt

inputs = tokenizer(formatted_prompt, return_tensors="np", padding=True)
input_ids = jnp.array(inputs["input_ids"])

# Shard the input across the batch dimension (replicated for single batch)
with mesh:
    input_sharding = NamedSharding(mesh, P(None, None))  # Replicate inputs
    input_ids = jax.device_put(input_ids, input_sharding)

# ----------------------------------------------------------------------
# 8. GENERATION LOOP
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("GENERATING...")
    print("="*80)

with mesh:
    current_ids = input_ids
    
    for step in range(MAX_NEW_TOKENS):
        outputs = model(current_ids, params=params)
        next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1, keepdims=True)
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        
        # Check for EOS
        if next_token[0, 0] == tokenizer.eos_token_id:
            break
        
        if jax.process_index() == 0 and step % 5 == 0:
            print(f"  Generated {step + 1} tokens...")

output_ids = current_ids

# ----------------------------------------------------------------------
# 9. DECODE AND PRINT OUTPUT
# ----------------------------------------------------------------------
# Gather output to host 0 for decoding
output_ids_np = jax.device_get(output_ids)
generated_text = tokenizer.decode(output_ids_np[0], skip_special_tokens=True)

if jax.process_index() == 0:
    print("\n" + "="*80)
    print("GENERATED OUTPUT")
    print("="*80)
    print(generated_text)
    print("="*80)