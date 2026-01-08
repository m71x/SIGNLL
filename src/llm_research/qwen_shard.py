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
DP_SIZE = 1     
FSDP_SIZE = 1   
TP_SIZE = 4    # All 32 chips working together
SP_SIZE = 1     

# ----------------------------------------------------------------------
# 1.5 INITIALIZE MESH BEFORE LOADING
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
# 3. LOAD MODEL WITH EASYDEL
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
    # [CRITICAL] I've uncommented this to ensure explicit sharding
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
print("GENERATED OUTPUT:")
print("="*80)
print(generated_text)

# ----------------------------------------------------------------------
# 7. DIAGNOSTICS & MEMORY LOGS (NEW SECTION)
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("FINAL SYSTEM DIAGNOSTICS")
print("="*80)

# A. Calculate Parameter Distribution
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
# bfloat16 is 2 bytes
total_model_gb = (total_params * 2) / 1e9 

print(f"Total Parameters:    {total_params / 1e9:.2f} Billion")
print(f"Total Model Size:    {total_model_gb:.2f} GB (bfloat16)")
print(f"Target per Chip:     {total_model_gb / 32:.2f} GB (Total / 32)")

# B. Check Actual Memory Usage on Local Devices
print("\nActual TPU Memory Usage (Local Devices):")
print("-" * 60)

local_devices = jax.local_devices()
# Limit print to first 4 devices to avoid spamming 32 lines
devices_to_show = local_devices[:4]

for i, device in enumerate(devices_to_show):
    try:
        # memory_stats returns dict with 'bytes_in_use'
        stats = device.memory_stats()
        used_gb = stats.get('bytes_in_use', 0) / 1e9
        limit_gb = stats.get('bytes_limit', 0) / 1e9
        
        # Calculate utilization %
        util_pct = (used_gb / limit_gb * 100) if limit_gb > 0 else 0.0
        
        print(f"Device {device.id} (Process {device.process_index}): "
              f"{used_gb:.2f} GB / {limit_gb:.2f} GB used ({util_pct:.1f}%)")
    except Exception as e:
        print(f"Device {device.id}: Could not fetch stats ({e})")

if len(local_devices) > 4:
    print(f"... (Stats hidden for remaining {len(local_devices)-4} devices)")

print("-" * 60)
print("NOTE: If 'Actual' is close to 'Target', sharding is working correctly.")
print("      Extra usage is due to KV Cache, JAX overhead, and buffers.")
print("="*80)