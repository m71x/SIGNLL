import os
import shutil

# =========================================================================
# CRITICAL CONFIGURATION (MUST BE BEFORE OTHER IMPORTS)
# =========================================================================
cache_dir = "/dev/shm/huggingface"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TMPDIR"] = cache_dir 
os.environ["TEMP"] = cache_dir
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

FLAGS = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 512,
}

def run_inference():
    xr.use_spmd()
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (num_devices,), ('model',))

    xm.rendezvous("cleanup_complete")

    # =========================================================================
    # DOWNLOAD PHASE
    # =========================================================================
    if xr.local_ordinal() == 0:
        print(f"Host {xr.global_ordinal()}: Checking/Downloading model...")
        try:
            AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
            AutoModelForCausalLM.from_pretrained(
                FLAGS["model_id"], 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            print(f"Host {xr.global_ordinal()}: Ready.")
        except Exception as e:
            print(f"Host {xr.global_ordinal()} Download Failed: {e}")
            raise e
    
    xm.rendezvous("download_done")

    # =========================================================================
    # LOADING PHASE
    # =========================================================================
    if xr.local_ordinal() == 0:
        print(f"Loading model from {cache_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
    
    # [FIX 1] Handle EOS Token List -> Single Int
    # Qwen often has multiple EOS tokens (e.g., <|endoftext|>, <|im_end|>). 
    # XLA crashes if we pass a list. We pick the first one (usually <|im_end|>)
    if isinstance(tokenizer.eos_token_id, list):
        # Pick the first one for stability
        stable_eos_token_id = tokenizer.eos_token_id[0] 
    else:
        stable_eos_token_id = tokenizer.eos_token_id

    # [FIX 2] Ensure Pad Token exists
    if tokenizer.pad_token is None:
        # If eos_token_id is a list, use the stable single int we found
        tokenizer.pad_token_id = stable_eos_token_id
        tokenizer.pad_token = tokenizer.decode(stable_eos_token_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        FLAGS["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model = model.to(device)

    # Apply Sharding
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            xs.mark_sharding(param, mesh, (0, None)) 
        else:
            xs.mark_sharding(param, mesh, (None,)) 

    # =========================================================================
    # INFERENCE PHASE
    # =========================================================================
    prompt = "Write a high-performance C++ implementation of a thread pool."
    
    # [FIX 3] Explicit Padding and Attention Mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,       
        truncation=True
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Replicate both input_ids and attention_mask
    xs.mark_sharding(input_ids, mesh, (None, None))
    xs.mark_sharding(attention_mask, mesh, (None, None))

    if xr.global_ordinal() == 0:
        print(f"Generating with EOS ID: {stable_eos_token_id}...")

    with torch.no_grad():
        # [FIX 4] Pass the SINGLE INTEGER eos_token_id
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=FLAGS["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stable_eos_token_id, # <--- Crucial Fix: Must be Int, not List
            do_sample=True,
            temperature=0.7
        )

    output_cpu = output.cpu()
    if xr.global_ordinal() == 0:
        print(f"\nRESPONSE:\n{tokenizer.decode(output_cpu[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference()