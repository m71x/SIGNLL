# convert_and_shard.py
# ============================================================================
# MUST COME FIRST — ENVIRONMENT SETUP
# ============================================================================
import os
import shutil

# Define the cache directory in shared memory (tmpfs)
CACHE_DIR = "/dev/shm/huggingface"

# Create the directory first
os.makedirs(CACHE_DIR, exist_ok=True)

# Check available space
stats = shutil.disk_usage("/dev/shm")
available_gb = stats.free / (1024**3)
print(f"Available space in /dev/shm: {available_gb:.2f} GB")

stats_root = shutil.disk_usage("/")
root_gb = stats_root.free / (1024**3)
print(f"Available space in /: {root_gb:.2f} GB")

if root_gb < 10:
    print(f"ERROR: Root filesystem is nearly full ({root_gb:.2f} GB free)")
    print(f"Will use /dev/shm exclusively for all operations")

# Set these BEFORE importing transformers/torch/jax
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"

# Control temporary/staging download locations
os.environ["TMPDIR"] = CACHE_DIR
os.environ["TEMP"] = CACHE_DIR
os.environ["TMP"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

# HuggingFace settings
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"

# JAX/TPU settings
os.environ["PJRT_DEVICE"] = "TPU"

# ============================================================================
# IMPORTS (Must be AFTER environment setup)
# ============================================================================
import jax
import jax.numpy as jnp
import numpy as np
import torch
import json
import ml_dtypes
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download

# ============================================================================
# CONFIG & MODEL LOADING
# ============================================================================
OUTPUT_DIR = "/home/mikexi/sharded_qwen32b" 
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

print("="*80)
print("STEP 1: Downloading model to /dev/shm first")
print(f"Cache location: {CACHE_DIR}")
print("="*80)

# CRITICAL FIX: Download model files explicitly to /dev/shm FIRST
# This bypasses the default download-to-root behavior
try:
    local_model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_dir=f"{CACHE_DIR}/models/{MODEL_ID.replace('/', '--')}",
    )
    print(f"✓ Model downloaded to: {local_model_path}")
except Exception as e:
    print(f"Download failed: {e}")
    print("Trying alternative method...")
    local_model_path = None

print("\n" + "="*80)
print("STEP 2: Loading PyTorch model into CPU memory")
print("="*80)

# Load config
if local_model_path:
    config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
else:
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)

# Load model - use local path if available, otherwise use cache_dir
model_source = local_model_path if local_model_path else MODEL_ID

model = AutoModelForCausalLM.from_pretrained(
    model_source,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
    local_files_only=bool(local_model_path),  # Don't re-download if we have local files
)
print(f"✓ Model loaded. Parameter count: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "="*80)
print("STEP 3: Converting to JAX-compatible arrays")
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
        
        # Cast to float32 first, then numpy, then ml_dtypes.bfloat16
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