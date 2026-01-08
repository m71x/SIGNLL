import jax
import sys
import os

# 1. CRITICAL: Initialize distributed system BEFORE importing EasyDeL
# This configures the backend to expect 8 workers instead of 1.
jax.distributed.initialize() 
from jax.experimental import multihost_utils
# 2. NOW it is safe to import EasyDeL
import easydel as ed

# Determine if this specific VM is the primary worker
is_master = jax.process_index() == 0

if is_master:
    print(f"Total devices: {jax.device_count()}")       # Should show 32
    print(f"Local devices: {jax.local_device_count()}") # Should show 4
    print("Starting model initialization...")

# ... (The rest of your code remains the same)
elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
    .set_dtype("bf16")
    .set_sharding(
        axis_dims=(1, 1, 8, 4, 1), 
        axis_names=("dp", "fsdp", "tp", "sp", "selective")
    )
    .set_esurge(max_model_len=4096, max_num_seqs=32)
)

esurge = elm.build_esurge()

for output in esurge.chat(
    [{"role": "user", "content": "Write a recursive fibonacci sequence implementation in python using memoization"}],
    sampling_params=ed.SamplingParams(max_tokens=512),
    stream=True,
):
    if is_master:
        print(output.delta_text, end="", flush=True)

if is_master:
    print(f"\nTokens/s: {output.tokens_per_second:.2f}")

# ---------------------------------------------------------
# #### NEW: Graceful Exit Block ####
# ---------------------------------------------------------
if is_master:
    print("\nWaiting for all workers to finish...")

# This creates a barrier. No worker can pass this line until 
# ALL 8 workers have reached it.
multihost_utils.sync_global_devices("shutting_down")

if is_master:
    print("All workers synced. Exiting safely.")

# Optional: Explicitly shutdown the distributed backend (good hygiene)
jax.distributed.shutdown()