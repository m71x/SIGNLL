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
from easydel import (
    AutoEasyDeLModelForCausalLM,
    EasyDeLState,
    PartitionAxis,
)

# --- UPDATED IMPORT LOGIC ---
try:
    from easydel.modules.qwen2 import create_qwen2_flax_partition_rules as qwen2_rules
except ImportError:
    # If the above fails, try the alternative naming convention
    try:
        from easydel.modules.qwen2 import qwen2_partition_rules as qwen2_rules
    except ImportError:
        qwen2_rules = None
# ----------------------------

from transformers import AutoTokenizer
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import json
import time

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

FSDP_SIZE = 8
TP_SIZE = 4
SP_SIZE = 1

# ----------------------------------------------------------------------
# 2. LOAD TOKENIZER
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
else:
    time.sleep(10)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------------
# 3. LOAD MODEL WITH EASYDEL
# ----------------------------------------------------------------------
devices = jax.devices()
device_array = mesh_utils.create_device_mesh((FSDP_SIZE, TP_SIZE, SP_SIZE), devices=devices)
mesh = Mesh(device_array, axis_names=('fsdp', 'tp', 'sp'))

partition_axis = PartitionAxis(
    fully_sharded_data_parallel_axis="fsdp", 
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

# Generate rules if available
if qwen2_rules is not None:
    partition_rules = qwen2_rules(partition_axis=partition_axis)
    if jax.process_index() == 0:
        print(f"✅ Qwen2 Partition Rules successfully imported.")
else:
    partition_rules = None
    if jax.process_index() == 0:
        print(f"⚠️  Could not find explicit partition rules. Falling back to Auto-Partitioning.")

# Load model
with mesh:
    result = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        partition_rules=partition_rules, # Will be None if import failed, triggering auto-mode
        auto_shard_params=True,
        sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
        sharding_axis_names=('fsdp', 'tp', 'sp'),
        partition_axis=partition_axis,
        shard_attention_computation=True,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        config_kwargs={"gradient_checkpointing": "", "use_scan_mlp": False}
    )

if isinstance(result, tuple):
    model, params = result
else:
    model = result
    params = model.params if hasattr(model, 'params') else None

# ----------------------------------------------------------------------
# 4. PARAMETER SHARDING ANALYSIS (YOUR ORIGINAL DEBUG LOGS)
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("PARAMETER SHARDING ANALYSIS")
    print("="*80)
    if params is not None:
        try:
            import jax.tree_util as tree_util
            leaves_with_paths, _ = tree_util.tree_flatten_with_path(params)
            
            print(f"\n🔍 INSPECTING FIRST 5 PARAMETERS FOR SHARDING:")
            for i, (path, val) in enumerate(leaves_with_paths[:5]):
                path_str = tree_util.keystr(path)
                print(f"\n  [{i}] {path_str}")
                if hasattr(val, 'sharding') and val.sharding is not None:
                    print(f"      Sharding type: {type(val.sharding).__name__}")
                    if hasattr(val.sharding, 'spec'):
                        print(f"      Sharding spec: {val.sharding.spec}")
                    if len(val.sharding.device_set) < 32:
                        print(f"      🚨 PROBLEM: Only on {len(val.sharding.device_set)}/32 devices!")
                else:
                    print(f"      🚨 NO SHARDING INFO FOUND")
        except Exception as e:
            print(f"❌ Error inspecting params: {e}")

# ----------------------------------------------------------------------
# 5. GENERATION
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."
if hasattr(tokenizer, "apply_chat_template"):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    formatted_prompt = prompt

inputs = tokenizer(formatted_prompt, return_tensors="jax", padding=False, truncation=True)
input_ids = inputs["input_ids"]

if jax.process_index() == 0:
    print("\nGENERATING RESPONSE...")

with mesh:
    current_ids = input_ids
    for step in range(MAX_NEW_TOKENS):
        outputs = model(current_ids)
        next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1, keepdims=True)
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        if next_token[0] == tokenizer.eos_token_id:
            break
    output_ids = current_ids

if jax.process_index() == 0:
    print("\n" + "="*80)
    print("GENERATED OUTPUT:")
    print("="*80)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print("="*80)