# ============================================================================
# MUST COME FIRST — ENVIRONMENT SETUP
# ============================================================================
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
os.environ["PJRT_DEVICE"] = "TPU"

# ============================================================================
# IMPORTS
# ============================================================================
import warnings
import torch
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
FLAGS = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 30,
}

# ============================================================================
# MAIN
# ============================================================================
def run_inference():
    xr.use_spmd()
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.arange(num_devices), (num_devices,), ("model",))

    xm.rendezvous("startup")

    # =========================================================================
    # DOWNLOAD (HOST 0 ONLY)
    # =========================================================================
    if xr.local_ordinal() == 0:
        print("Downloading model to /dev/shm ...")

        AutoTokenizer.from_pretrained(
            FLAGS["model_id"],
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        AutoModelForCausalLM.from_pretrained(
            FLAGS["model_id"],
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        print("Download complete.")

    xm.rendezvous("download_complete")

    # =========================================================================
    # LOAD
    # =========================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        FLAGS["model_id"],
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    eos_token_id = (
        tokenizer.eos_token_id[0]
        if isinstance(tokenizer.eos_token_id, list)
        else tokenizer.eos_token_id
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = eos_token_id
        tokenizer.pad_token = tokenizer.decode(eos_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        FLAGS["model_id"],
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 🔑 CRITICAL OPTIMIZATION: Use Static Cache
    # This prevents the "recompile-every-step" issue by allocating a fixed-size
    # KV cache graph that XLA can optimize once.
    model.generation_config.cache_implementation = "static"
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device)

    # =========================================================================
    # SHARDING
    # =========================================================================
    for _, param in model.named_parameters():
        if param.dim() >= 2:
            xs.mark_sharding(param, mesh, (0, None))
        else:
            xs.mark_sharding(param, mesh, (None,))

    # =========================================================================
    # INPUT
    # =========================================================================
    prompt = "Write a high-performance C++ implementation of a thread pool."

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(torch.int32).to(device)

    xs.mark_sharding(input_ids, mesh, (None, None))
    xs.mark_sharding(attention_mask, mesh, (None, None))

    # =========================================================================
    # GENERATE (OPTIMIZED)
    # =========================================================================
    if xr.global_ordinal() == 0:
        print("Compiling and Generating (Static Cache)...")

    # The first run will trigger one large compilation (which may take a moment),
    # but subsequent tokens will generate rapidly without recompiling.
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=FLAGS["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,  # Greedy decoding
        use_cache=True
    )

    # Trigger execution
    xm.mark_step()

    # =========================================================================
    # OUTPUT
    # =========================================================================
    # Move to CPU for decoding
    output_cpu = output_ids.cpu()

    if xr.global_ordinal() == 0:
        print("\nRESPONSE:\n")
        print(tokenizer.decode(output_cpu[0], skip_special_tokens=True))


if __name__ == "__main__":
    run_inference()