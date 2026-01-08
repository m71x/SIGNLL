# convert_and_shard.py
# ============================================================================
# MUST COME FIRST — ENVIRONMENT SETUP
# ============================================================================
import os

# Define the cache directory in shared memory (tmpfs)
CACHE_DIR = "/dev/shm/huggingface"

# Set these BEFORE importing transformers/torch/jax
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
# IMPORTS (Must be AFTER environment setup)
# ============================================================================
import jax
import jax.numpy as jnp
import numpy as np
import torch
import json
import ml_dtypes  # Required for bfloat16 support in numpy
from transformers import AutoModelForCausalLM, AutoConfig

# ============================================================================
# CONFIG & MODEL LOADING
# ============================================================================
OUTPUT_DIR = "/home/mikexi/sharded_qwen32b" 
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

print("="*80)
print("STEP 1: Loading PyTorch model into CPU memory")
print(f"Cache location: {os.environ.get('HF_HOME', 'Not Set')}")
print("="*80)

# Load config
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
print("STEP 2: Converting to JAX-compatible arrays")
print("="*80)

def get_sharding_spec(name: str) -> tuple:
    """Return PartitionSpec axes for each parameter type."""
    if "embed_tokens" in name or "lm_head" in name:
        return ("fsdp", "tp")
    elif any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]):
        return ("fsdp", "tp")
    elif any(proj in name for proj in ["o_proj", "down_proj"]):
        return ("tp", "fsdp")
    elif any(proj in name for proj in ["gate_proj", "up_proj"]):
        return ("fsdp", "tp")
    elif "norm" in name:
        return ("fsdp",)
    else:
        return ("fsdp", "tp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save config
config.save_pretrained(OUTPUT_DIR)
print(f"✓ Config saved to {OUTPUT_DIR}")

params_index = {}

# Use torch.no_grad to ensure no graph overhead during conversion
with torch.no_grad():
    for name, param in model.named_parameters():
        cpu_tensor = param.detach().cpu()
        
        # FIX: Cast to float32 first, then numpy, then ml_dtypes.bfloat16
        # PyTorch bfloat16 -> numpy fails. 
        # PyTorch bfloat16 -> PyTorch float32 -> numpy -> ml_dtypes.bfloat16 works.
        arr = cpu_tensor.float().numpy().astype(ml_dtypes.bfloat16)
        
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
with open(os.path.join(OUTPUT_DIR, "params_index.json"), "w") as f:
    json.dump(params_index, f, indent=2)

print(f"\n✓ All parameters saved to {OUTPUT_DIR}")
print(f"✓ Index saved to {OUTPUT_DIR}/params_index.json")

# Cleanup to free memory immediately
del model
import gc
gc.collect()

print("\n" + "="*80)
print("CONVERSION COMPLETE")
print("="*80)