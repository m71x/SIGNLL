import jax
import sys
import os
import gc # Added for memory cleanup

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
    for output in esurge.chat(
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
if is_master:
    print("\n[Cleanup] destroying engine references...")
del esurge
del elm
gc.collect() # Force Python to release the C++ backend objects

# 2. GLOBAL BARRIER
#    We wait here to ensure EVERYONE has finished the loop 
#    and cleaned up their local objects.
if is_master:
    print("Waiting for all workers to sync before termination...")

multihost_utils.sync_global_devices("ready_to_kill")

# 3. THE ATOMIC EXIT
#    Do NOT call jax.distributed.shutdown(). 
#    We just verified everyone is ready to die via the barrier above.
#    Now we pull the plug simultaneously.
if is_master:
    print("Sync complete. forcing immediate process termination.")
    sys.stdout.flush() # Ensure logs appear

# Force kill the process. 
# This skips Python cleanup and ignores hung non-daemon threads.
os._exit(0)