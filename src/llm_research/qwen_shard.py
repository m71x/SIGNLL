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
# IMPORTS
# ============================================================================
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from easydel import AutoEasyDeLModelForCausalLM, PartitionAxis
from transformers import AutoTokenizer, GenerationConfig, AutoConfig

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
MAX_NEW_TOKENS = 1900

# TPU MESH CONFIGURATION (32 Chips)
# [CRITICAL] For a single prompt (Batch=1), FSDP/DP MUST be 1.
DP_SIZE = 1     
FSDP_SIZE = 1   
TP_SIZE = 32    # All 32 chips working together on the single prompt
SP_SIZE = 1     

# ----------------------------------------------------------------------
# 1.5 [CRITICAL] INITIALIZE MESH BEFORE LOADING
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("INITIALIZING JAX MESH")
print("="*80)
total_devices = jax.device_count()
print(f"Total JAX Devices Visible: {total_devices}")

if total_devices != 32:
    print(f"⚠️ WARNING: TP_SIZE=32 but JAX sees {total_devices} devices.")

# Create the mesh immediately
device_mesh = mesh_utils.create_device_mesh((DP_SIZE, FSDP_SIZE, TP_SIZE, SP_SIZE))
mesh = Mesh(device_mesh, axis_names=('dp', 'fsdp', 'tp', 'sp'))
print(f"✓ Mesh created: {mesh}")

# ----------------------------------------------------------------------
# 2. LOAD TOKENIZER & CONFIG
# ----------------------------------------------------------------------
print("\nLoading Tokenizer and Config...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

CONTEXT_LENGTH = 2048
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
config.max_position_embeddings = CONTEXT_LENGTH

# ----------------------------------------------------------------------
# 3. LOAD MODEL WITH EASYDEL (PASSING MESH)
# ----------------------------------------------------------------------
print(f"\nLoading Model with EasyDeL...")

# Load model with corrected sharding axes AND the active mesh
result = AutoEasyDeLModelForCausalLM.from_pretrained(
    MODEL_ID,
    config_kwargs={
        "max_position_embeddings": CONTEXT_LENGTH,
        "max_sequence_length": CONTEXT_LENGTH, 
        "gradient_checkpointing": "",
        "use_scan_mlp": False,
    },
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=jax.lax.Precision.DEFAULT,
    # [CRITICAL] Pass the mesh explicitly here
    #mesh=mesh,  
    sharding_axis_dims=(DP_SIZE, FSDP_SIZE, TP_SIZE, SP_SIZE),
    sharding_axis_names=('dp', 'fsdp', 'tp', 'sp'), 
    partition_axis=PartitionAxis(),
    shard_attention_computation=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

if isinstance(result, tuple):
    model, params = result
else:
    model = result
    params = model.params if hasattr(model, 'params') else None

print("✓ Model loaded!")

# ----------------------------------------------------------------------
# 4. PREPARE INPUT
# ----------------------------------------------------------------------
prompt = "Write a python program that shows an example of the diamond problem in inheritance."

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

# Pad remainder
current_len = input_ids.shape[1]
predicted_total_len = current_len + MAX_NEW_TOKENS
remainder = predicted_total_len % SP_SIZE
if remainder != 0:
    pad_amt = SP_SIZE - remainder
    pad_ids = jnp.full((input_ids.shape[0], pad_amt), tokenizer.pad_token_id, dtype=input_ids.dtype)
    pad_mask = jnp.zeros((attention_mask.shape[0], pad_amt), dtype=attention_mask.dtype)
    input_ids = jnp.concatenate([pad_ids, input_ids], axis=1)
    attention_mask = jnp.concatenate([pad_mask, attention_mask], axis=1)

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
)
if not hasattr(generation_config, 'forced_decoder_ids'):
    generation_config.forced_decoder_ids = None

print("Sharding inputs to Mesh...")
input_sharding = NamedSharding(mesh, P()) 

# Device put using the mesh created in Step 1.5
input_ids = jax.device_put(input_ids, input_sharding)
attention_mask = jax.device_put(attention_mask, input_sharding)

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

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n" + "="*80)
print(generated_text)
print("="*80)