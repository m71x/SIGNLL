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
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from easydel import (
    AutoEasyDeLModelForCausalLM,
    PartitionAxis,
)
from transformers import AutoTokenizer, GenerationConfig, AutoConfig

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 50

# [CRITICAL] LIMIT CONTEXT WINDOW
# We restrict the model to 2048 tokens to prevent OOM errors.
# This saves memory by shrinking the Attention Mask from ~1GB to ~4MB.
MAX_CONTEXT_LENGTH = 2048 

# TPU MESH CONFIGURATION (32 Chips)
DP_SIZE = 1     
FSDP_SIZE = 1   
TP_SIZE = 8     # Tensor Parallel
SP_SIZE = 4     # Sequence Parallel

# ----------------------------------------------------------------------
# 2. LOAD TOKENIZER & CONFIGURE MODEL
# ----------------------------------------------------------------------
print("Loading Tokenizer and Config...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Config and FORCE smaller context
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
config.max_position_embeddings = MAX_CONTEXT_LENGTH
config.sliding_window = None # Disable sliding window to simplify memory usage
print(f"✓ Forced max_position_embeddings to: {config.max_position_embeddings}")

# ----------------------------------------------------------------------
# 3. LOAD MODEL WITH EASYDEL
# ----------------------------------------------------------------------
print(f"\nLoading Model with EasyDeL...")
print(f"Mesh Configuration: DP={DP_SIZE}, FSDP={FSDP_SIZE}, TP={TP_SIZE}, SP={SP_SIZE}")

# Load model with Explicit Input Shape
# This input_shape argument is CRITICAL to prevent JAX from allocating the full 128k buffer.
result = AutoEasyDeLModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    input_shape=(1, MAX_CONTEXT_LENGTH), # [FIX] Force static shape compilation
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=jax.lax.Precision.DEFAULT,
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

# [FIX] PAD BASED ON TOTAL LENGTH
# We pad so that (Input + Output) matches our fixed MAX_CONTEXT_LENGTH context window?
# Actually, for SP=4 compatibility, we just need the CURRENT input to be divisible by 4.
# But since we fixed the model shape to MAX_CONTEXT_LENGTH, we should pad appropriately.
# For simplicity in this test, we just pad to be divisible by SP_SIZE.

current_len = input_ids.shape[1]
remainder = current_len % SP_SIZE

if remainder != 0:
    pad_amt = SP_SIZE - remainder
    print(f"Padding input (len {current_len}) by {pad_amt} tokens for SP compatibility.")
    
    pad_ids = jnp.full((input_ids.shape[0], pad_amt), tokenizer.pad_token_id, dtype=input_ids.dtype)
    pad_mask = jnp.zeros((attention_mask.shape[0], pad_amt), dtype=attention_mask.dtype)
    
    # Left padding
    input_ids = jnp.concatenate([pad_ids, input_ids], axis=1)
    attention_mask = jnp.concatenate([pad_mask, attention_mask], axis=1)

print(f"Final Input shape: {input_ids.shape}")

# ----------------------------------------------------------------------
# 5. GENERATION
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("GENERATING RESPONSE...")
print("="*80)

generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
)

if not hasattr(generation_config, 'forced_decoder_ids'):
    generation_config.forced_decoder_ids = None

print("Creating JAX Mesh context...")
device_mesh = mesh_utils.create_device_mesh((DP_SIZE, FSDP_SIZE, TP_SIZE, SP_SIZE))
mesh = Mesh(device_mesh, axis_names=('dp', 'fsdp', 'tp', 'sp'))

print("Starting generation loop inside Mesh context...")

with mesh:
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

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