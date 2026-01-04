import os
import shutil
import warnings
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================================
# CONFIGURATION
# =========================================================================
# 1. Force standard HTTP download (Fixes "CAS service error" / "IO Error")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

# 2. Use RAM disk for storage to avoid filling root disk
cache_dir = "/dev/shm/huggingface"
os.environ["HF_HOME"] = cache_dir

# 3. Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

FLAGS = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 512,
}

def clean_failed_runs():
    """Attempts to clean the cache directory if it exists to free up space."""
    # Only Rank 0 on each physical VM should clean up
    if xr.local_ordinal() == 0:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass

def run_inference():
    # Initialize TPU environment
    xr.use_spmd()
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    
    # Define the mesh for 32 chips
    # We shard on the 'model' axis
    mesh = Mesh(np.array(range(num_devices)), (num_devices,), ('model',))

    # Clean up previous failed runs
    clean_failed_runs()
    xm.rendezvous("cleanup_complete")

    # =========================================================================
    # DOWNLOAD PHASE (One process per Host)
    # =========================================================================
    if xr.local_ordinal() == 0:
        print(f"Host {xr.global_ordinal()}: Downloading {FLAGS['model_id']} to RAM ({cache_dir})...")
        try:
            AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
            AutoModelForCausalLM.from_pretrained(
                FLAGS["model_id"], 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            print(f"Host {xr.global_ordinal()}: Download complete.")
        except Exception as e:
            print(f"Host {xr.global_ordinal()} Download Failed: {e}")
            raise e
    
    # Sync: Wait for all hosts to finish downloading
    xm.rendezvous("download_done")

    # =========================================================================
    # LOADING PHASE (All Processes)
    # =========================================================================
    if xr.local_ordinal() == 0:
        print(f"Loading model from {cache_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
    
    # Load model to CPU memory first (using low_cpu_mem_usage to avoid spikes)
    # REMOVED: with xs.sharding_context(mesh): <-- This caused the error
    model = AutoModelForCausalLM.from_pretrained(
        FLAGS["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Move to TPU device (Lazy execution starts here)
    # XLA won't fully materialize the 32B parameters on HBM until we step or mark sharding
    model = model.to(device)

    # Apply Sharding IMMEDIATELY
    # This instructs the compiler to split the weights across the mesh
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            # Shard the first dimension (usually output features) across the 32 devices
            xs.mark_sharding(param, mesh, (0, -1)) 
        else:
            # Replicate 1D tensors (biases/layer norms)
            xs.mark_sharding(param, mesh, (None,)) 

    # =========================================================================
    # INFERENCE PHASE
    # =========================================================================
    prompt = "Write a high-performance C++ implementation of a thread pool."
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # Replicate input across all devices (Tensor Parallelism requirement)
    xs.mark_sharding(input_ids, mesh, (None, None))

    if xr.global_ordinal() == 0:
        print("Generating...")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, 
            max_new_tokens=FLAGS["max_new_tokens"]
        )

    # Gather and print (Master only)
    output_cpu = output.cpu()
    if xr.global_ordinal() == 0:
        print(f"\nRESPONSE:\n{tokenizer.decode(output_cpu[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference()