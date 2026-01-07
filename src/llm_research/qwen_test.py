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
    
    # Try multiple methods to inspect parameters
    if params is not None:
        print("\n🔍 INSPECTING PARAMETER STRUCTURE...")
        
        try:
            import jax.tree_util as tree_util
            
            # Get all parameter leaves
            leaves = tree_util.tree_leaves(params)
            print(f"  Found {len(leaves)} parameter tensors via tree_leaves")
            
            total_params = 0
            total_bytes = 0
            
            for leaf in leaves:
                if hasattr(leaf, 'shape'):
                    param_count = int(jnp.prod(jnp.array(leaf.shape)))
                    param_bytes = param_count * 2
                    total_params += param_count
                    total_bytes += param_bytes
            
            print(f"\n📊 PARAMETER STATISTICS:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Total size: {total_bytes / (1024**3):.2f} GB (bfloat16)")
            print(f"  Per-device (ideal): {total_bytes / (1024**3) / 32:.2f} GB")
            
            # Get structure with paths - FIXED
            leaves_with_paths, treedef = tree_util.tree_flatten_with_path(params)
            
            print(f"\n🔍 FIRST 15 PARAMETERS:")
            for i, (path, val) in enumerate(leaves_with_paths[:15]):
                if hasattr(val, 'shape') and len(val.shape) > 0:  # Skip scalars
                    size_mb = (jnp.prod(jnp.array(val.shape)) * 2) / (1024**2)
                    path_str = tree_util.keystr(path)
                    print(f"\n  [{i}] {path_str}")
                    print(f"      Shape: {val.shape}")
                    print(f"      Size: {size_mb:.2f} MB")
                    print(f"      Dtype: {val.dtype}")
                    
                    if hasattr(val, 'sharding') and val.sharding is not None:
                        sharding_type = type(val.sharding).__name__
                        print(f"      Sharding type: {sharding_type}")
                        
                        if hasattr(val.sharding, 'spec'):
                            print(f"      Sharding spec: {val.sharding.spec}")
                            print(f"      Mesh: {val.sharding.mesh.axis_names}")
                        elif 'SingleDevice' in sharding_type:
                            print(f"      🚨 SINGLE DEVICE SHARDING - NOT DISTRIBUTED!")
                        else:
                            print(f"      Unknown sharding: {val.sharding}")
                        
                        try:
                            device_set = val.sharding.device_set
                            print(f"      Device count: {len(device_set)}")
                            if len(device_set) == 1:
                                print(f"      🚨 ON ONLY 1 DEVICE!")
                        except:
                            pass
                    else:
                        print(f"      ⚠️  NO SHARDING INFO")
            
            print(f"\n🔍 LAST 15 WEIGHT PARAMETERS (skipping RNG states):")
            weight_params = [(path, val) for path, val in leaves_with_paths 
                           if hasattr(val, 'shape') and len(val.shape) > 0 and 'weight' in tree_util.keystr(path).lower()]
            
            for i, (path, val) in enumerate(weight_params[-15:]):
                size_mb = (jnp.prod(jnp.array(val.shape)) * 2) / (1024**2)
                path_str = tree_util.keystr(path)
                print(f"\n  [{len(weight_params)-15+i}] {path_str}")
                print(f"      Shape: {val.shape}")
                print(f"      Size: {size_mb:.2f} MB")
                print(f"      Dtype: {val.dtype}")
                
                if hasattr(val, 'sharding') and val.sharding is not None:
                    sharding_type = type(val.sharding).__name__
                    print(f"      Sharding type: {sharding_type}")
                    
                    if hasattr(val.sharding, 'spec'):
                        print(f"      Sharding spec: {val.sharding.spec}")
                        print(f"      Mesh: {val.sharding.mesh.axis_names}")
                    elif 'SingleDevice' in sharding_type:
                        print(f"      🚨 SINGLE DEVICE SHARDING - NOT DISTRIBUTED!")
                    
                    try:
                        device_set = val.sharding.device_set
                        print(f"      Device count: {len(device_set)}")
                        if len(device_set) < 32:
                            print(f"      ⚠️  Only on {len(device_set)}/32 devices!")
                    except:
                        pass
                else:
                    print(f"      ⚠️  NO SHARDING INFO")
            
            # Find largest parameters
            print(f"\n🚨 20 LARGEST PARAMETERS:")
            params_with_sizes = [
                (tree_util.keystr(path), val, jnp.prod(jnp.array(val.shape)) * 2)
                for path, val in leaves_with_paths 
                if hasattr(val, 'shape') and len(val.shape) > 0
            ]
            params_with_sizes.sort(key=lambda x: x[2], reverse=True)
            
            for path_str, val, size_bytes in params_with_sizes[:20]:
                size_mb = size_bytes / (1024**2)
                print(f"\n  {path_str}")
                print(f"    Shape: {val.shape}")
                print(f"    Size: {size_mb:.2f} MB")
                print(f"    Dtype: {val.dtype}")
                
                if hasattr(val, 'sharding') and val.sharding is not None:
                    sharding_type = type(val.sharding).__name__
                    print(f"    Sharding type: {sharding_type}")
                    
                    if hasattr(val.sharding, 'spec'):
                        print(f"    Sharding spec: {val.sharding.spec}")
                        print(f"    Mesh: {val.sharding.mesh.axis_names}")
                    elif 'SingleDevice' in sharding_type:
                        print(f"    🚨 SINGLE DEVICE SHARDING!")
                    
                    try:
                        device_set = val.sharding.device_set
                        num_devices = len(device_set)
                        print(f"    Device count: {num_devices}/32")
                        if num_devices < 32:
                            print(f"    🚨🚨🚨 PROBLEM: Only on {num_devices} devices, not all 32!")
                    except:
                        pass
                else:
                    print(f"    🚨 NO SHARDING - REPLICATED ON ALL DEVICES!")
                    
        except Exception as e:
            print(f"❌ Error inspecting params: {e}")
            import traceback
            traceback.print_exc()

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