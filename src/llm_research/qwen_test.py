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
# IMPORTS (AFTER ENV VARS)
# ============================================================================
import jax
import jax.numpy as jnp
from easydel import (
    AutoEasyDeLModelForCausalLM,
    EasyDeLState,
    PartitionAxis,
)
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# TPU MESH CONFIGURATION (32 Chips)
# Adjusting for better memory distribution: more FSDP, less TP
FSDP_SIZE = 8  # Fully Sharded Data Parallel (increased)
TP_SIZE = 4    # Tensor Parallelism (decreased)
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
print(f"Mesh Configuration: FSDP={FSDP_SIZE}, TP={TP_SIZE}, SP={SP_SIZE}")
print(f"Available devices: {len(jax.devices())}")

# Load model with automatic sharding
result = AutoEasyDeLModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=jax.lax.Precision.DEFAULT,
    sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
    sharding_axis_names=('fsdp', 'tp', 'sp'),
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
print(f"✓ Model type: {type(model)}")
print(f"✓ Params available: {params is not None}")

# ----------------------------------------------------------------------
# 4. PREPARE INPUT
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."

# Apply chat template if available
if hasattr(tokenizer, "apply_chat_template"):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print(f"\nFormatted prompt:\n{formatted_prompt}\n")
else:
    formatted_prompt = prompt

# Tokenize
inputs = tokenizer(
    formatted_prompt, 
    return_tensors="jax",
    padding=False,
    truncation=True,
)

input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", jnp.ones_like(input_ids))

print(f"Input shape: {input_ids.shape}")
print(f"Input tokens: {input_ids.shape[1]}")

# ----------------------------------------------------------------------
# 5. GENERATION
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("GENERATING RESPONSE...")
print("="*80)

# Create a proper generation config
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
)

# Generate with the model
try:
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )
except Exception as e:
    print(f"Generation with config failed: {e}")
    print("Trying simpler generation...")
    # Fallback to manual generation
    outputs = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
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

# Statistics
num_new_tokens = len(output_ids[0]) - len(input_ids[0])
print(f"\n✓ Generated {num_new_tokens} new tokens")
print(f"✓ Total tokens: {len(output_ids[0])}")