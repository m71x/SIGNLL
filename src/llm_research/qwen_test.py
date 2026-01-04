import os
import shutil
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================================
# CONFIGURATION & ENV SETUP
# =========================================================================

# 1. Force standard HTTP download (Fixes "CAS service error" / "IO Error")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

# 2. Use RAM disk for storage
# NOTE: /dev/shm is local to each host.
cache_dir = "/dev/shm/huggingface"
os.environ["HF_HOME"] = cache_dir

FLAGS = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 512,
}

def clean_failed_runs():
    """Attempts to clean the cache directory if it exists to free up space."""
    if xm.get_local_ordinal() == 0:
        # Only check this once per host
        if os.path.exists(cache_dir):
            # Check usage. If near full or corrupted from crash, wipe it.
            # For safety in this script, we assume if we are restarting, we want a clean slate
            # to avoid the 'No space left' error.
            try:
                print(f"Host {xm.get_ordinal()}: Cleaning previous cache at {cache_dir}...")
                shutil.rmtree(cache_dir)
            except Exception as e:
                print(f"Warning: Could not clear cache: {e}")

def run_inference():
    xr.use_spmd()
    device = xm.xla_device()
    
    # 32 TPU devices setup
    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (num_devices,), ('model',))

    # Clean up previous failed runs (One worker per host does this)
    clean_failed_runs()
    
    # Wait for cleanup to finish on all chips
    xm.rendezvous("cleanup_complete")

    # =========================================================================
    # DOWNLOAD PHASE (One process per Host)
    # =========================================================================
    # Critical Fix: Use get_local_ordinal() so ONE worker per PHYSICAL HOST downloads.
    # The previous `is_master_ordinal()` only downloaded on Host 0, leaving Host 1+ empty.
    if xm.get_local_ordinal() == 0:
        print(f"Host {xm.get_ordinal()} (Local Rank 0): Downloading {FLAGS['model_id']} to RAM...")
        try:
            # Download Tokenizer
            AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
            # Download Model (just to populate cache)
            AutoModelForCausalLM.from_pretrained(
                FLAGS["model_id"], 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            print(f"Host {xm.get_ordinal()}: Download complete.")
        except Exception as e:
            print(f"Host {xm.get_ordinal()} Download Failed: {e}")
            raise e
    
    # Sync: Wait for all hosts to finish downloading
    xm.rendezvous("download_done")

    # =========================================================================
    # LOADING PHASE (All Processes)
    # =========================================================================
    print(f"Process {xm.get_ordinal()}: Loading model from local RAM cache...")
    
    tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"])
    
    # Use sharding context to spread weights across TPUs immediately
    # This prevents OOM by not loading the full model on CPU first
    with xs.sharding_context(mesh):
        model = AutoModelForCausalLM.from_pretrained(
            FLAGS["model_id"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    # Explicitly shard large weight matrices row-wise
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            xs.mark_sharding(param, mesh, (0, -1)) 
        else:
            xs.mark_sharding(param, mesh, (None,)) 

    model = model.to(device)

    # =========================================================================
    # INFERENCE PHASE
    # =========================================================================
    prompt = "Write a high-performance C++ implementation of a thread pool."
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # Replicate input across all devices (Standard for Tensor Parallelism)
    xs.mark_sharding(input_ids, mesh, (None, None))

    if xm.is_master_ordinal():
        print("Generating...")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, 
            max_new_tokens=FLAGS["max_new_tokens"]
        )

    # Gather and print (Master only)
    if xm.is_master_ordinal():
        print(f"\nRESPONSE:\n{tokenizer.decode(output[0].cpu(), skip_special_tokens=True)}")

if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference()