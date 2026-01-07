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

from easydel import AutoEasyDeLModelForCausalLM, PartitionAxis
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# TPU MESH CONFIGURATION (32 Chips)
FSDP_SIZE = 8
TP_SIZE = 4
SP_SIZE = 1

# ----------------------------------------------------------------------
# 2. LOAD TOKENIZER
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
# 3. DEVICE AND MESH SETUP
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("DEVICE AND MESH SETUP")
    print("="*80)

# Use ALL devices across ALL hosts (global mesh)
all_devices = jax.devices()
total_devices = len(all_devices)
expected_devices = FSDP_SIZE * TP_SIZE * SP_SIZE

if jax.process_index() == 0:
    print(f"Total global devices: {total_devices}")
    print(f"Expected devices: {expected_devices}")
    print(f"FSDP={FSDP_SIZE}, TP={TP_SIZE}, SP={SP_SIZE}")

if total_devices != expected_devices:
    raise RuntimeError(
        f"Device count mismatch: got {total_devices}, expected {expected_devices}"
    )

device_array = mesh_utils.create_device_mesh(
    (FSDP_SIZE, TP_SIZE, SP_SIZE),
    devices=all_devices,
)

mesh = Mesh(device_array, axis_names=('fsdp', 'tp', 'sp'))

if jax.process_index() == 0:
    print(f"Mesh shape: {mesh.devices.shape}")
    print(f"Mesh axis names: {mesh.axis_names}")
    print(f"Total devices in mesh: {mesh.devices.size}")

partition_axis = PartitionAxis(
    fully_sharded_data_parallel_axis="fsdp",
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

# ----------------------------------------------------------------------
# 4. LOAD MODEL
# ----------------------------------------------------------------------
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)

with mesh:
    result = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
        sharding_axis_names=('fsdp', 'tp', 'sp'),
        partition_axis=partition_axis,
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
# 5. PARAMETER SHARDING ANALYSIS
# ----------------------------------------------------------------------
if jax.process_index() == 0 and params is not None:
    print("\n" + "="*80)
    print("PARAMETER SHARDING ANALYSIS")
    print("="*80)

    import jax.tree_util as tree_util
    leaves_with_paths, _ = tree_util.tree_flatten_with_path(params)

    print(f"\n🔍 CHECKING CRITICAL WEIGHTS:")
    for path, val in leaves_with_paths:
        path_str = tree_util.keystr(path)
        if "layers" in path_str and "down_proj" in path_str and "kernel" in path_str:
            print(f"\n  Weight: {path_str}")
            print(f"    Shape: {val.shape}")
            sharding_type = type(val.sharding).__name__
            print(f"    Sharding Type: {sharding_type}")
            if hasattr(val.sharding, "spec"):
                print(f"    Spec: {val.sharding.spec}")
            break

# ----------------------------------------------------------------------
# 6. GENERATION
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

with mesh:
    input_sharding = NamedSharding(mesh, P(None, None))
    input_ids = jax.device_put(input_ids, input_sharding)
    current_ids = input_ids
    
    for _ in range(MAX_NEW_TOKENS):
        outputs = model(current_ids, params=params)
        next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1, keepdims=True)
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        if next_token[0, 0] == tokenizer.eos_token_id:
            break

output_ids = current_ids
output_ids_np = jax.device_get(output_ids)
generated_text = tokenizer.decode(output_ids_np[0], skip_special_tokens=True)

if jax.process_index() == 0:
    print("\n" + "="*80)
    print("GENERATED OUTPUT")
    print("="*80)
    print(generated_text)
    print("="*80)