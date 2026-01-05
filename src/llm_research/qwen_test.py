import os

CACHE_DIR = "/dev/shm/huggingface"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"

os.environ["TMPDIR"] = CACHE_DIR
os.environ["TEMP"] = CACHE_DIR
os.environ["TMP"] = CACHE_DIR

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import shutil
import warnings
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# =========================================================================
# CONFIGURATION
# =========================================================================


warnings.filterwarnings("ignore")

FLAGS = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 512,
}

# [FIX] Custom Stopping Criteria for TPU
# Bypasses torch.isin which crashes on XLA with type errors
class TPUStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Simple equality check is stable on XLA
        return input_ids[:, -1] == self.eos_token_id

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
        print(f"Loading model from {CACHE_DIR}...")
    
    tokenizer = AutoTokenizer.from_pretrained(FLAGS["model_id"], trust_remote_code=True)
    
    # 1. Determine Stable EOS Token
    if isinstance(tokenizer.eos_token_id, list):
        stable_eos_token_id = tokenizer.eos_token_id[0]
    else:
        stable_eos_token_id = tokenizer.eos_token_id

    # 2. Fix Padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = stable_eos_token_id
        tokenizer.pad_token = tokenizer.decode(stable_eos_token_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        FLAGS["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 3. Prevent Default EOS Logic (which uses torch.isin)
    # We must clear this in the config so generate() doesn't auto-add the crashing criteria
    model.generation_config.eos_token_id = None

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
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,       
        truncation=True
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    xs.mark_sharding(input_ids, mesh, (None, None))
    xs.mark_sharding(attention_mask, mesh, (None, None))

    # [FIX] Use Custom Criteria
    tpu_stopping_criteria = StoppingCriteriaList([
        TPUStoppingCriteria(stable_eos_token_id)
    ])

    if xr.global_ordinal() == 0:
        print(f"Generating with Custom TPU Criteria (EOS: {stable_eos_token_id})...")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=FLAGS["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=tpu_stopping_criteria, # <--- Use our custom class
            # eos_token_id=None,                     # <--- Do NOT pass this (uses default crashing logic)
            do_sample=True,
            temperature=0.7
        )

    output_cpu = output.cpu()
    if xr.global_ordinal() == 0:
        print(f"\nRESPONSE:\n{tokenizer.decode(output_cpu[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference()