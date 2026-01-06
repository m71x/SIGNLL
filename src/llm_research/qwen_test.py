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

# Performance Tuning Flags
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PJRT_DEVICE"] = "TPU"

# ============================================================================
# IMPORTS
# ============================================================================
import jax
import jax.numpy as jnp
from jax.sharding import Mesh           # [FIX] Essential for manual mesh context
from jax.experimental import mesh_utils # [FIX] Essential for creating device mesh
from easydel import (
    AutoEasyDeLModelForCausalLM,
    EasyDeLState,
    PartitionAxis,
)
from transformers import AutoTokenizer, GenerationConfig

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# [FIX] TPU MESH CONFIGURATION (32 Chips)
# We use a 4D mesh to satisfy the model's requirement for a 'dp' axis.
# Total devices must equal product: 1 * 8 * 4 * 1 = 32
DP_SIZE = 1    # Data Parallel (Added to fix "axis 'dp' not found")
FSDP_SIZE = 8  # Fully Sharded Data Parallel
TP_SIZE = 4    # Tensor Parallelism
SP_SIZE = 1    # Sequence Parallelism

# ----------------------------------------------------------------------
# 2. LOAD TOKENIZER
# ----------------------------------------------------------------------
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------------
# 3. LOAD MODEL WITH EASYDEL
# ----------------------------------------------------------------------
print(f"\nLoading Model with EasyDeL...")
print(f"Mesh Configuration: DP={DP_SIZE}, FSDP={FSDP_SIZE}, TP={TP_SIZE}, SP={SP_SIZE}")
print(f"Available devices: {len(jax.devices())}")

# Load model with corrected sharding axes
result = AutoEasyDeLModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=jax.lax.Precision.DEFAULT,
    # [FIX] Update axis dims and names to include 'dp'
    sharding_axis_dims=(DP_SIZE, FSDP_SIZE, TP_SIZE, SP_SIZE),
    sharding_axis_names=('dp', 'fsdp', 'tp', 'sp'), 
    partition_axis=PartitionAxis(),
    shard_attention_computation=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
    config_kwargs={
        "gradient_checkpointing": "",
        "use_scan_mlp": False,
    }
)

# Handle different return types
if isinstance(result, tuple):
    model, params = result
else:
    model = result
    params = model.params if hasattr(model, 'params') else None

print("✓ Model loaded and sharded successfully!")

# ----------------------------------------------------------------------
# 4. PREPARE INPUT
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
attention_mask = inputs.get("attention_mask", jnp.ones_like(input_ids))

print(f"Input shape: {input_ids.shape}")

# ----------------------------------------------------------------------
# 5. GENERATION
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("GENERATING RESPONSE...")
print("="*80)

# [FIX] Configure Generation properly
generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
)

# [FIX] Patch for transformers/EasyDeL compatibility
# Newer transformers removed these attributes, but EasyDeL might still check them
if not hasattr(generation_config, 'forced_decoder_ids'):
    generation_config.forced_decoder_ids = None
if not hasattr(generation_config, 'forced_bos_token_id'):
    generation_config.forced_bos_token_id = None
if not hasattr(generation_config, 'forced_eos_token_id'):
    generation_config.forced_eos_token_id = None

# [FIX] CREATE MESH AND ENTER CONTEXT
# We explicitly create the mesh matching the sharding_axis_names defined above
print("Creating JAX Mesh context...")
device_mesh = mesh_utils.create_device_mesh((DP_SIZE, FSDP_SIZE, TP_SIZE, SP_SIZE))
mesh = Mesh(device_mesh, axis_names=('dp', 'fsdp', 'tp', 'sp'))

print("Starting generation loop inside Mesh context...")

with mesh:
    # We use the high-level generate function which handles the loop
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

# Extract generated sequences
if hasattr(outputs, 'sequences'):
    output_ids = outputs.sequences
else:
    output_ids = outputs

# ----------------------------------------------------------------------
# 6. DECODE AND DISPLAY OUTPUT
# ----------------------------------------------------------------------
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n" + "="*80)
print("GENERATED OUTPUT:")
print("="*80)
print(generated_text)
print("="*80)