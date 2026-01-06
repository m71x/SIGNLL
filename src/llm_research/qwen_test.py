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

# ============================================================================
# INITIALIZE DISTRIBUTED JAX FOR MULTI-HOST TPU
# ============================================================================
# This is critical for TPU pods with multiple hosts
jax.distributed.initialize()

print(f"JAX initialized:")
print(f"  Process index: {jax.process_index()}")
print(f"  Process count: {jax.process_count()}")
print(f"  Local devices: {jax.local_device_count()}")
print(f"  Global devices: {jax.device_count()}")

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
# 2. LOAD TOKENIZER (Only on process 0 to avoid duplicate downloads)
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
    import time
    time.sleep(10)
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
print(f"Total devices needed: {FSDP_SIZE * TP_SIZE * SP_SIZE}")

# Create PartitionAxis with proper parameters matching the mesh
partition_axis = PartitionAxis(
    data_parallel_axis="dp",
    fully_sharded_data_parallel_axis="fsdp", 
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

# Only load from disk on process 0
if jax.process_index() == 0:
    print("Process 0: Loading model from disk...")

# Load model with automatic sharding across all hosts
result = AutoEasyDeLModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=jax.lax.Precision.DEFAULT,
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

# Handle different return types
if isinstance(result, tuple):
    model, params = result
else:
    model = result
    params = model.params if hasattr(model, 'params') else None

if jax.process_index() == 0:
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

# Create a proper generation config with all needed attributes
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

# Add missing attributes that EasyDeL expects
if not hasattr(generation_config, 'forced_decoder_ids'):
    generation_config.forced_decoder_ids = None
if not hasattr(generation_config, 'forced_bos_token_id'):
    generation_config.forced_bos_token_id = None
if not hasattr(generation_config, 'forced_eos_token_id'):
    generation_config.forced_eos_token_id = None

# Generate with the model
try:
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )
except Exception as e:
    print(f"Generation with full config failed: {e}")
    print("Trying simplest generation...")
    # Fallback: use model's default generation config
    try:
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    except Exception as e2:
        print(f"Simple generation also failed: {e2}")
        print("Trying manual decoding loop...")
        # Last resort: manual generation
        current_ids = input_ids
        for _ in range(MAX_NEW_TOKENS):
            logits = model(current_ids).logits
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            current_ids = jnp.concatenate([current_ids, next_token], axis=1)
            if next_token[0] == tokenizer.eos_token_id:
                break
        outputs = type('obj', (object,), {'sequences': current_ids})()

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