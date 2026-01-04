import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use /dev/shm for RAM-based storage
os.environ["HF_HOME"] = "/dev/shm/huggingface"

FLAGS = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 512,
}

def run_inference():
    xr.use_spmd()
    device = xm.xla_device()
    
    # 32 TPU devices setup
    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (num_devices,), ('model',))

    # Barrier: Only master downloads to RAM, others wait
    if xm.is_master_ordinal():
        print(f"Master downloading {FLAGS['model_id']} to RAM (/dev/shm)...")
        AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(FLAGS["model_id"], torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    xm.rendezvous("download_done") # All workers wait here

    # Load from the RAM cache created by master
    tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"])
    
    # Use sharding context to spread across all 32 TPUs during load
    with xs.sharding_context(mesh):
        model = AutoModelForCausalLM.from_pretrained(
            FLAGS["model_id"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    # Explicitly shard large weight matrices row-wise across the 32-device mesh
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            xs.mark_sharding(param, mesh, (0, -1)) 
        else:
            xs.mark_sharding(param, mesh, (None,)) 

    model = model.to(device)

    # Inference
    prompt = "Write a high-performance C++ implementation of a thread pool."
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # Replicate input across all devices (Standard for Tensor Parallelism)
    xs.mark_sharding(input_ids, mesh, (None, None))

    if xm.is_master_ordinal():
        print("Generating...")

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=FLAGS["max_new_tokens"])

    if xm.is_master_ordinal():
        print(f"\nRESPONSE:\n{tokenizer.decode(output[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference()