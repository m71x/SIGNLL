import jax
import sys
import os
import gc # Added for memory cleanup
from jax import numpy as jnp
from transformers import AutoTokenizer
from jax import lax
# 1. CRITICAL: Initialize distributed system BEFORE importing EasyDeL
jax.distributed.initialize() 
from jax.experimental import multihost_utils

# 2. NOW it is safe to import EasyDeL
import easydel as ed


# Determine if this specific VM is the primary worker
is_master = jax.process_index() == 0

if is_master:
    print(f"Total devices: {jax.device_count()}")
    print(f"Local devices: {jax.local_device_count()}")
    print("Starting model initialization...")

model_id = "your-model-id"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=lax.Precision.DEFAULT,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3,
        attn_dtype=jnp.bfloat16,
    ),
)

# Initialize eSurge engine
engine = ed.eSurge(
    model=model,
    tokenizer=tokenizer,
    max_model_len=8192,
    max_num_seqs=16,
    hbm_utilization=0.9,
    page_size=64,
)
# ---------------------------------------------------------
# #### NEW: Define your list of independent prompts ####
# ---------------------------------------------------------
prompts = [
    "Write a recursive fibonacci sequence implementation in python using memoization",
    "Explain the difference between TCP and UDP in one paragraph.",
    "Write a haiku about a TPU pod."
]

# ---------------------------------------------------------
# #### NEW: Loop through prompts ####
# ---------------------------------------------------------
for i, user_prompt in enumerate(prompts):
    
    # Master prints a separator to keep logs clean
    if is_master:
        print(f"\n\n{'='*40}")
        print(f"PROMPT {i+1}: {user_prompt}")
        print(f"{'='*40}\nResponse: ", end="", flush=True)

    # We create a FRESH conversation list for every loop iteration.
    # This ensures no history (and no KV cache) is carried over from the previous run.
    conversation = [{"role": "user", "content": user_prompt}]

    # Run the chat generation
    for output in engine.chat(
        conversation,
        sampling_params=ed.SamplingParams(max_tokens=512),
        stream=True,
    ):
        if is_master:
            print(output.delta_text, end="", flush=True)

    if is_master:
        print(f"\n\n[Stats] Tokens/s: {output.tokens_per_second:.2f}")

# ---------------------------------------------------------
# Graceful Exit Block
# ---------------------------------------------------------
# ---------------------------------------------------------
# Graceful Exit Block
# ---------------------------------------------------------

# 1. Clean up the engine explicitly to free device memory
#    (Helps prevent "Resource Busy" errors on the next run)
