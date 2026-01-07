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
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import json

if jax.process_index() == 0:
    print("\n" + "="*80)
    print("DEVICE AND MESH SETUP")
    print("="*80)

# Get all devices
devices = jax.devices()

if jax.process_index() == 0:
    print(f"\n📊 DEVICE INFORMATION:")
    print(f"  Total global devices: {jax.device_count()}")
    print(f"  Local devices: {jax.local_device_count()}")
    print(f"  Process index: {jax.process_index()}")
    print(f"  Process count: {jax.process_count()}")
    
    print(f"\n📋 DEVICE LIST:")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
        print(f"    - Platform: {device.platform}")
        print(f"    - ID: {device.id}")
        print(f"    - Process index: {device.process_index}")

print(f"\nLoading Model with EasyDeL...")
print(f"Mesh Configuration: FSDP={FSDP_SIZE}, TP={TP_SIZE}, SP={SP_SIZE}")
print(f"Total devices needed: {FSDP_SIZE * TP_SIZE * SP_SIZE}")

# Create the device mesh
device_array = mesh_utils.create_device_mesh((FSDP_SIZE, TP_SIZE, SP_SIZE), devices=devices)
mesh = Mesh(device_array, axis_names=('fsdp', 'tp', 'sp'))

if jax.process_index() == 0:
    print(f"\n🔷 MESH CREATED:")
    print(f"  Mesh shape: {mesh.shape}")
    print(f"  Axis names: {mesh.axis_names}")
    print(f"  Device mesh shape: {device_array.shape}")
    print(f"  Mesh:\n{mesh}")

# Create PartitionAxis WITHOUT 'dp' axis (only use axes in mesh)
partition_axis = PartitionAxis(
    fully_sharded_data_parallel_axis="fsdp", 
    tensor_parallel_axis="tp",
    sequence_parallel_axis="sp",
)

if jax.process_index() == 0:
    print(f"\n✅ PartitionAxis configured")
    print(f"  FSDP axis: fsdp (size={FSDP_SIZE})")
    print(f"  TP axis: tp (size={TP_SIZE})")
    print(f"  SP axis: sp (size={SP_SIZE})")

# Only load from disk on process 0
if jax.process_index() == 0:
    print("\n" + "="*80)
    print("MODEL LOADING")
    print("="*80)
    print("Process 0: Loading model from disk...")

# Load model with automatic sharding across all hosts
with mesh:
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
    print("\n" + "="*80)
    print("PARAMETER SHARDING ANALYSIS")
    print("="*80)
    print(f"✓ Model type: {type(model)}")
    print(f"✓ Params available: {params is not None}")
    print(f"✓ Params type: {type(params)}")
    
    if params is not None:
        # Handle different param types
        try:
            # Try to convert to dict-like structure
            if hasattr(params, 'unfreeze'):
                # FrozenDict
                params_dict = params.unfreeze()
            elif hasattr(params, 'to_pure_dict'):
                # NNX State
                params_dict = params.to_pure_dict()
            elif hasattr(params, '__dict__'):
                params_dict = params.__dict__
            else:
                params_dict = dict(params)
            
            from flax.traverse_util import flatten_dict
            flat_params = flatten_dict(params_dict, sep='.')
        except Exception as e:
            print(f"  Warning: Could not flatten params: {e}")
            print(f"  Attempting alternative inspection...")
            # Alternative: iterate directly
            flat_params = {}
            
            def collect_params(prefix, obj):
                if hasattr(obj, 'shape'):
                    flat_params[prefix] = obj
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        collect_params(f"{prefix}.{k}" if prefix else k, v)
                elif hasattr(obj, '__dict__'):
                    for k, v in obj.__dict__.items():
                        collect_params(f"{prefix}.{k}" if prefix else k, v)
            
            collect_params('', params)
        
        print(f"\n📊 PARAMETER STATISTICS:")
        total_params = 0
        total_bytes = 0
        param_count_by_type = {}
        
        # Count parameters
        for key, value in flat_params.items():
            if hasattr(value, 'shape'):
                param_count = int(jnp.prod(jnp.array(value.shape)))
                param_bytes = param_count * 2  # bfloat16 = 2 bytes
                total_params += param_count
                total_bytes += param_bytes
                
                # Track by layer type
                layer_type = key.split('.')[0] if '.' in key else 'other'
                param_count_by_type[layer_type] = param_count_by_type.get(layer_type, 0) + param_bytes
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total size: {total_bytes / (1024**3):.2f} GB (bfloat16)")
        print(f"  Per-device (ideal): {total_bytes / (1024**3) / 32:.2f} GB")
        print(f"  Number of param tensors: {len(flat_params)}")
        
        print(f"\n📈 SIZE BY COMPONENT:")
        for comp, size in sorted(param_count_by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {comp}: {size / (1024**3):.2f} GB")
        
        print(f"\n🔍 SHARDING DETAILS (Sample of parameters):")
        # Show first few, middle, and last few parameters
        keys = list(flat_params.keys())
        sample_keys = keys[:10] + keys[len(keys)//2:len(keys)//2+5] + keys[-10:]
        
        for key in sample_keys:
            value = flat_params[key]
            if hasattr(value, 'shape'):
                param_size_mb = (jnp.prod(jnp.array(value.shape)) * 2) / (1024**2)
                print(f"\n  {key}:")
                print(f"    Shape: {value.shape}")
                print(f"    Size: {param_size_mb:.2f} MB")
                
                if hasattr(value, 'sharding') and value.sharding is not None:
                    print(f"    Sharding spec: {value.sharding.spec}")
                    print(f"    Sharding mesh: {value.sharding.mesh.axis_names}")
                    
                    # Check which devices this parameter is on
                    try:
                        device_set = value.sharding.device_set
                        print(f"    Spread across: {len(device_set)} devices")
                    except Exception as e:
                        print(f"    Could not determine device set: {e}")
                else:
                    print(f"    ⚠️  NO SHARDING INFO - might be replicated!")
        
        # Critical check: look for largest parameters
        print(f"\n🚨 LARGEST PARAMETERS (potential OOM culprits):")
        param_sizes = [(k, v, jnp.prod(jnp.array(v.shape)) * 2 if hasattr(v, 'shape') else 0) 
                       for k, v in flat_params.items()]
        param_sizes.sort(key=lambda x: x[2], reverse=True)
        
        for key, value, size_bytes in param_sizes[:15]:
            size_mb = size_bytes / (1024**2)
            print(f"\n  {key}:")
            print(f"    Shape: {value.shape}")
            print(f"    Size: {size_mb:.2f} MB")
            if hasattr(value, 'sharding') and value.sharding is not None:
                print(f"    Sharding: {value.sharding.spec}")
                try:
                    device_set = value.sharding.device_set
                    print(f"    Devices: {len(device_set)}")
                except:
                    pass
            else:
                print(f"    ⚠️  NO SHARDING - REPLICATED ACROSS ALL DEVICES!")

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

# Generate within the mesh context
with mesh:
    # Manual generation loop (most reliable for EasyDeL)
    current_ids = input_ids
    
    for step in range(MAX_NEW_TOKENS):
        if jax.process_index() == 0 and step % 10 == 0:
            print(f"Generating token {step}/{MAX_NEW_TOKENS}...")
        
        # Forward pass
        outputs = model(current_ids)
        logits = outputs.logits
        
        # Get next token (greedy decoding for simplicity)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        
        # Append to sequence
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        
        # Check for EOS
        if next_token[0] == tokenizer.eos_token_id:
            if jax.process_index() == 0:
                print("EOS token generated, stopping...")
            break
    
    output_ids = current_ids

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