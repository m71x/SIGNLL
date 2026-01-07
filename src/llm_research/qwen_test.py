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

# This MUST come before importing easydel or calling any JAX operations
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
from transformers import AutoTokenizer
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import json
import time
import re
from dataclasses import dataclass
from typing import Optional, Union, Tuple

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# TPU MESH CONFIGURATION (32 Chips)
FSDP_SIZE = 8  # Fully Sharded Data Parallel
TP_SIZE = 4    # Tensor Parallelism
SP_SIZE = 1    # Sequence Parallelism

# ----------------------------------------------------------------------
# 2. DEFINITIONS (FIX FOR MISSING IMPORTS)
# ----------------------------------------------------------------------

# We define PartitionRule locally to avoid ImportError hell
@dataclass
class PartitionRule:
    primary: Optional[str]
    fsdp: Optional[Union[str, Tuple[str, ...]]]

# ----------------------------------------------------------------------
# 3. LOAD TOKENIZER
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
else:
    # Wait for process 0 to download
    time.sleep(10)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------------
# 4. LOAD MODEL WITH MANUAL PARTITION RULES
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("DEVICE AND MESH SETUP")
    print("="*80)

devices = jax.devices()
device_array = mesh_utils.create_device_mesh((FSDP_SIZE, TP_SIZE, SP_SIZE), devices=devices)
mesh = Mesh(device_array, axis_names=('fsdp', 'tp', 'sp'))

partition_axis = PartitionAxis(
    fully_sharded_data_parallel_axis="fsdp", 
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

# ======================================================================
# MANUALLY DEFINE SHARDING RULES
# Using our local PartitionRule class
# ======================================================================
partition_rules = (
    # Embeddings
    (re.compile(".*embed_tokens.*"), PartitionRule(None, ("fsdp", "tp"))),
    
    # Attention Q, K, V
    (re.compile(".*q_proj.*"),      PartitionRule(None, ("fsdp", "tp"))),
    (re.compile(".*k_proj.*"),      PartitionRule(None, ("fsdp", "tp"))),
    (re.compile(".*v_proj.*"),      PartitionRule(None, ("fsdp", "tp"))),
    
    # Attention Output
    (re.compile(".*o_proj.*"),      PartitionRule(None, ("tp", "fsdp"))),
    
    # MLP
    (re.compile(".*gate_proj.*"),   PartitionRule(None, ("fsdp", "tp"))),
    (re.compile(".*up_proj.*"),     PartitionRule(None, ("fsdp", "tp"))),
    (re.compile(".*down_proj.*"),   PartitionRule(None, ("tp", "fsdp"))),
    
    # Output Head
    (re.compile(".*lm_head.*"),     PartitionRule(None, ("fsdp", "tp"))),
    
    # Layer Norms
    (re.compile(".*norm.*"),        PartitionRule(None, ("fsdp",))),
    
    # Catch-all
    (re.compile(".*"),              PartitionRule(None, ("fsdp", "tp"))),
)

if jax.process_index() == 0:
    print(f"\n✅ Manual Partition Rules Applied (Locally Defined).")

# Load model using the manual rules
with mesh:
    result = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        # PASS THE MANUAL RULES
        partition_rules=partition_rules,
        auto_shard_params=True,
        sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
        sharding_axis_names=('fsdp', 'tp', 'sp'),
        partition_axis=partition_axis,
        shard_attention_computation=True,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        config_kwargs={
            "gradient_checkpointing": "",
            "use_scan_mlp": False,
        }
    )

if jax.process_index() == 0:
    print("✓ Model loaded successfully across all hosts!")

if isinstance(result, tuple):
    model, params = result
else:
    model = result
    params = model.params if hasattr(model, 'params') else None

# ----------------------------------------------------------------------
# 5. INSPECTION
# ----------------------------------------------------------------------
if jax.process_index() == 0 and params is not None:
    print("\n" + "="*80)
    print("PARAMETER SHARDING ANALYSIS")
    print("="*80)
    
    try:
        import jax.tree_util as tree_util
        leaves_with_paths, _ = tree_util.tree_flatten_with_path(params)
        
        # Check specific weights to ensure they are NOT SingleDeviceSharding
        print(f"\n🔍 CHECKING CRITICAL WEIGHTS:")
        for path, val in leaves_with_paths:
            path_str = tree_util.keystr(path)
            # Check an MLP layer (previously crashing)
            if "layers" in path_str and "down_proj" in path_str and "kernel" in path_str:
                print(f"\n  Weight: {path_str}")
                print(f"    Shape: {val.shape}")
                if hasattr(val, 'sharding'):
                    sharding_type = type(val.sharding).__name__
                    print(f"    Sharding Type: {sharding_type}")
                    if hasattr(val.sharding, 'spec'):
                        print(f"    Spec: {val.sharding.spec}")
                    
                    if 'SingleDevice' in sharding_type:
                        print("    🚨 STILL FAILING TO SHARD!")
                    elif 'NamedSharding' in sharding_type:
                        print("    ✅ SUCCESS: NamedSharding active")
                break
    except Exception as e:
        print(f"Inspection error: {e}")

# ----------------------------------------------------------------------
# 6. GENERATION
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."

if hasattr(tokenizer, "apply_chat_template"):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
else:
    formatted_prompt = prompt

inputs = tokenizer(
    formatted_prompt, 
    return_tensors="jax",
    padding=False,
    truncation=True,
)

input_ids = inputs["input_ids"]

if jax.process_index() == 0:
    print("\n" + "="*80)
    print("GENERATING RESPONSE...")
    print("="*80)

with mesh:
    current_ids = input_ids
    for step in range(MAX_NEW_TOKENS):
        if jax.process_index() == 0 and step % 5 == 0:
            print(f"Generating token {step}/{MAX_NEW_TOKENS}...")
        
        outputs = model(current_ids)
        next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1, keepdims=True)
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    output_ids = current_ids

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

if jax.process_index() == 0:
    print("\n" + "="*80)
    print("GENERATED OUTPUT:")
    print("="*80)
    print(generated_text)
    print("="*80)