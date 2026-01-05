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
# IMPORTS (AFTER ENV VARS)
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
    "max_new_tokens": 1,
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

    # Stable EOS
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

    # CRITICAL: Disable all dynamic stopping logic
    model.generation_config.eos_token_id = None
    model.generation_config.use_cache = False

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
    # INFERENCE (STATIC DECODE ONLY)
    # =========================================================================
    prompt = "Write a high-performance C++ implementation of a thread pool."

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    xs.mark_sharding(input_ids, mesh, (None, None))
    xs.mark_sharding(attention_mask, mesh, (None, None))

    if xr.global_ordinal() == 0:
        print("Generating (TPU-safe static decoding)...")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=FLAGS["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,

            # TPU CRITICAL FLAGS
            use_cache=False,
            synced_gpus=False,
        )

    # =========================================================================
    # POST-PROCESS (CPU EOS TRIM)
    # =========================================================================
    output_cpu = output.cpu()

    if eos_token_id is not None:
        eos_pos = (output_cpu[0] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            output_cpu = output_cpu[:, : eos_pos[0] + 1]

    if xr.global_ordinal() == 0:
        print("\nRESPONSE:\n")
        print(tokenizer.decode(output_cpu[0], skip_special_tokens=True))


if __name__ == "__main__":
    run_inference()
