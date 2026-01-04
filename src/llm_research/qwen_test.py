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
    # 1. Enable SPMD Mode (Crucial for sharding)
    xr.use_spmd()
    
    device = xm.xla_device()
    
    # 2. Setup Device Mesh
    # We create a 1D mesh across all 32 devices
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices,)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('model',))

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)

    # 4. Load Model (Initialize on CPU, then shard to TPU)
    if xm.is_master_ordinal():
        print(f"Loading {FLAGS['model_id']} and sharding across {num_devices} devices...")

    # Load onto CPU first to avoid OOM during the 'to(device)' call
    model = AutoModelForCausalLM.from_pretrained(
        FLAGS["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 5. Apply Sharding to Model Parameters
    # This tells XLA to split the weights across the 'model' axis of our mesh
    for name, param in model.named_parameters():
        # Shard large weight matrices, replicate small biases/scalars
        if param.dim() >= 2:
            xs.mark_sharding(param, mesh, (0, -1)) # Shard row-wise
        else:
            xs.mark_sharding(param, mesh, (None,)) # Replicate

    model = model.to(device)

    # 6. Inference logic
    prompt = "Write a high-performance C++ implementation of a thread pool."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Shard the input tensor (batch dim)
    xs.mark_sharding(inputs.input_ids, mesh, (0, 1))

    if xm.is_master_ordinal():
        print("Generating...")

    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=FLAGS["max_new_tokens"],
            do_sample=True,
            temperature=0.7
        )

    # 7. Decode and Print (Master only)
    if xm.is_master_ordinal():
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nRESPONSE:\n{response}")

if __name__ == "__main__":
    # In SPMD mode, we don't use xmp.spawn. 
    # We run the script directly and XLA handles the distribution.
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference()