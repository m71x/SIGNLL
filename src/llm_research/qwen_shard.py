# convert_and_shard.py
# ============================================================================
# MUST COME FIRST — ENVIRONMENT SETUP
# ============================================================================
import os
import shutil

# 1. Define the RAM disk location
# /dev/shm is a standard Linux directory mapped to shared memory (RAM)
CACHE_DIR = "/dev/shm/huggingface"

# 2. Cleanup old cache first to ensure space
if os.path.exists(CACHE_DIR):
    try:
        shutil.rmtree(CACHE_DIR)
        print(f"Cleaned up old cache at {CACHE_DIR}")
    except Exception as e:
        print(f"Warning: Could not clean old cache: {e}")

# 3. Create the directory
os.makedirs(CACHE_DIR, exist_ok=True)

# 4. Check available space in RAM
stats = shutil.disk_usage("/dev/shm")
available_gb = stats.free / (1024**3)
print(f"Available space in /dev/shm: {available_gb:.2f} GB")

if available_gb < 70:  # 32B model needs ~65GB+ for raw weights + overhead
    print("WARNING: You might not have enough RAM (/dev/shm) for a 32B model.")
    print("Ensure you have at least 70GB+ of system RAM available.")

# 5. Set Environment Variables BEFORE importing transformers
# This overrides the default ~/.cache/huggingface location
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"

# Redirect temporary files to RAM as well
os.environ["TMPDIR"] = CACHE_DIR
os.environ["TEMP"] = CACHE_DIR
os.environ["TMP"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

# HuggingFace optimization settings
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
os.environ["PJRT_DEVICE"] = "TPU"

# ============================================================================
# IMPORTS
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
print("STEP 1: Downloading model explicitly to /dev/shm")
print(f"Target Location: {CACHE_DIR}")
print("="*80)

# Explicitly download to /dev/shm using snapshot_download
# local_dir_use_symlinks=False ensures real files are written to /dev/shm, not symlinks to ~/.cache
try:
    local_model_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=f"{CACHE_DIR}/models/{MODEL_ID.replace('/', '--')}",
        local_dir_use_symlinks=False,  # CRITICAL: Force actual files in RAM
        cache_dir=CACHE_DIR,
    )
    print(f"✓ Model successfully downloaded to RAM: {local_model_path}")
except Exception as e:
    print(f"CRITICAL ERROR: Download failed: {e}")
    exit(1)

print("\n" + "="*80)
print("STEP 2: Loading PyTorch model from RAM")
print("="*80)

# Load config from the local path in RAM
config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)

# Load model using the local path
# local_files_only=True ensures we don't accidentally touch the network/root cache
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True, 
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

# Cleanup
del model
import gc
gc.collect()
# Optional: Free up the RAM immediately after use
try:
    shutil.rmtree(CACHE_DIR)
    print(f"✓ Cleared RAM cache at {CACHE_DIR}")
except:
    pass

print("\n" + "="*80)
print("CONVERSION COMPLETE")
print("="*80)