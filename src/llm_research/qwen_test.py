import os
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
FLAGS = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 512,
}

def run_inference():
    # 1. Enable SPMD Mode
    xr.use_spmd()
    
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    
    # 2. Setup Device Mesh (1D Mesh across all 32 devices)
    mesh_shape = (num_devices,)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('model',))

    # 3. Load Tokenizer (Master downloads, others wait)
    # Using a barrier ensures only rank 0 downloads/caches first.
    if xm.is_master_ordinal():
        tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
    
    # Wait for master to finish downloading tokenizer
    xm.rendezvous("tokenizer_download_complete")
    
    if not xm.is_master_ordinal():
        # Others load from the cache populated by master
        tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)

    # 4. Load Model (Optimized for Sharding)
    if xm.is_master_ordinal():
        print(f"Loading {FLAGS['model_id']} and sharding across {num_devices} devices...")

    # We use a barrier again to ensure only master downloads the model weights to cache.
    # This prevents 32 processes from hammering the HF Hub simultaneously.
    if xm.is_master_ordinal():
        _ = AutoModelForCausalLM.from_pretrained(
            FLAGS["model_id"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # We just want to trigger the download/cache here, not keep the object
        )
    xm.rendezvous("model_download_complete")

    # 5. Define Sharding Rules
    # We define the partition spec *before* fully loading to device if possible,
    # or use the sharding_context to help distribute the load.
    
    # NOTE: 'low_cpu_mem_usage=True' uses the 'accelerate' library backend to 
    # load weights on CPU meta device -> materialize -> move to device.
    # To shard *immediately* across TPUs without holding full model in CPU RAM,
    # we can use the XLA SPMD sharding context.
    
    print(f"Process {xm.get_ordinal()} loading model...")
    
    # This context manager applies the sharding automatically as layers are initialized/moved
    with xs.sharding_context(mesh):
        # We assume the model fits in CPU RAM or is handled by accelerate's offloading.
        # Since we are using low_cpu_mem_usage=True, it loads meta tensors first.
        model = AutoModelForCausalLM.from_pretrained(
            FLAGS["model_id"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=None # Important: Let XLA handle device placement
        )

    # Manual Sharding Application (Reinforcement)
    # Sometimes auto-sharding needs explicit hints for weights
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            xs.mark_sharding(param, mesh, (0, -1)) # Shard row-wise
        else:
            xs.mark_sharding(param, mesh, (None,)) # Replicate

    # Move to device (If not already handled by context)
    model = model.to(device)

    # 6. Inference logic
    prompt = "Write a high-performance C++ implementation of a thread pool."
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Mark sharding on inputs
    # Input ID shape is [Batch, SeqLen]. We replicate or shard batch.
    # Since batch=1, we usually replicate or shard dim 0 (if batch > 1).
    # For text generation, replicating inputs across the model mesh is often safer 
    # unless you are doing Data Parallelism. Here we are doing Tensor Parallelism (sharding weights).
    xs.mark_sharding(input_ids, mesh, (None, None)) 
    xs.mark_sharding(attention_mask, mesh, (None, None))

    if xm.is_master_ordinal():
        print("Generating...")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=FLAGS["max_new_tokens"],
            do_sample=True,
            temperature=0.7
        )

    # 7. Decode and Print (Master only)
    # Gather output from device 0 (or all if sharded results)
    # generated tokens are usually replicated in this setup
    output_cpu = output.cpu()
    
    if xm.is_master_ordinal():
        response = tokenizer.decode(output_cpu[0], skip_special_tokens=True)
        print(f"\nRESPONSE:\n{response}")

if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference()