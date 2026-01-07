# convert_and_shard.py
# Run this ONCE on a single worker to pre-shard the checkpoint

import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import numpy as np

# Don't initialize distributed - we're running on single worker
os.environ["PJRT_DEVICE"] = "TPU"

CACHE_DIR = "/dev/shm/huggingface"
OUTPUT_DIR = "/home/mikexi/sharded_qwen32b"  # Change this!
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Sharding config (must match your TPU pod setup)
FSDP_SIZE = 8
TP_SIZE = 4
SP_SIZE = 1

print("="*80)
print("STEP 1: Loading PyTorch model into CPU memory")
print("="*80)

import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Load config first
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load model in CPU with streaming to reduce peak memory
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
print(f"✓ Model loaded. Parameter count: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "="*80)
print("STEP 2: Converting to JAX arrays with sharding specs")
print("="*80)

# Create the mesh for sharding specification
devices = jax.devices()
print(f"Available devices: {len(devices)}")

# For conversion, we'll define the target sharding but store unsharded
# The sharding will be applied when loading on multi-host

def get_sharding_spec(name: str) -> tuple:
    """Return (shape suffix, PartitionSpec) for each parameter type."""
    if "embed_tokens" in name or "lm_head" in name:
        return ("fsdp", "tp")
    elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
        return ("fsdp", "tp")
    elif "o_proj" in name or "down_proj" in name:
        return ("tp", "fsdp")
    elif "gate_proj" in name or "up_proj" in name:
        return ("fsdp", "tp")
    elif "norm" in name:
        return ("fsdp",)
    else:
        return ("fsdp", "tp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save config
config.save_pretrained(OUTPUT_DIR)
print(f"✓ Config saved to {OUTPUT_DIR}")

# Convert and save each parameter
params_index = {}
for name, param in model.named_parameters():
    # Convert to numpy (bf16)
    arr = param.detach().cpu().to(torch.bfloat16).numpy()
    
    # Save to file
    safe_name = name.replace(".", "_")
    filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.npy")
    np.save(filepath, arr)
    
    # Record metadata
    params_index[name] = {
        "file": f"{safe_name}.npy",
        "shape": list(arr.shape),
        "dtype": "bfloat16",
        "sharding": get_sharding_spec(name),
    }
    print(f"  Saved: {name} -> {arr.shape}")

# Save index file
import json
with open(os.path.join(OUTPUT_DIR, "params_index.json"), "w") as f:
    json.dump(params_index, f, indent=2)

print(f"\n✓ All parameters saved to {OUTPUT_DIR}")
print(f"✓ Index saved to {OUTPUT_DIR}/params_index.json")

# Cleanup
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n" + "="*80)
print("CONVERSION COMPLETE")
print("="*80)
print(f"Output directory: {OUTPUT_DIR}")
print("Now run the inference script on your multi-host TPU pod.")