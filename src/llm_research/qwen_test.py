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
# INITIALIZE JAX DISTRIBUTED
# ============================================================================
import jax
jax.distributed.initialize()

print(f"JAX distributed initialized:")
print(f"  Process index: {jax.process_index()}")
print(f"  Process count: {jax.process_count()}")
print(f"  Local devices: {jax.local_device_count()}")
print(f"  Global devices: {jax.device_count()}")

# ============================================================================
# IMPORTS
# ============================================================================
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import time

from easydel import AutoEasyDeLModelForCausalLM, PartitionAxis
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30
FSDP_SIZE = 8
TP_SIZE = 4
SP_SIZE = 1

# ----------------------------------------------------------------------
# LOAD TOKENIZER
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
# CREATE GLOBAL MESH
# ----------------------------------------------------------------------
all_devices = jax.devices()
device_array = mesh_utils.create_device_mesh((FSDP_SIZE, TP_SIZE, SP_SIZE), devices=all_devices)
mesh = Mesh(device_array, axis_names=('fsdp', 'tp', 'sp'))

partition_axis = PartitionAxis(
    fully_sharded_data_parallel_axis="fsdp",
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

if jax.process_index() == 0:
    print(f"Global mesh: {mesh.devices.shape} = {mesh.devices.size} devices")

# ----------------------------------------------------------------------
# LOAD MODEL WITH 8-BIT QUANTIZATION (KEY FIX)
# ----------------------------------------------------------------------
with mesh:
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        # ⬇️ KEY: 8-bit quantization reduces loading memory by ~4x
        load_in_8bit=True,
        # Sharding configuration
        sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
        sharding_axis_names=('fsdp', 'tp', 'sp'),
        partition_axis=partition_axis,
        auto_shard_params=True,
        shard_attention_computation=True,
        input_shape=(1, 1),
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        config_kwargs={"use_scan_mlp": False},
    )

if jax.process_index() == 0:
    print("✓ Model loaded successfully!")

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
    print("\n" + "="*60)
    print("OUTPUT:")
    print("="*60)
    print(generated_text)