# reset_all_checkpoints.py

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from checkpoint_access import get_core_ordinal, reset_checkpoint_samples


def run_for_core(index):
    """
    Function to be launched once per TPU core.
    Each core will reset its own checkpoint samples independently.
    """
    core_id = get_core_ordinal()
    print(f"[Core {core_id}] 🔄 Starting checkpoint reset...", flush=True)
    
    success = reset_checkpoint_samples(core_id)
    if success:
        print(f"[Core {core_id}] ✅ Checkpoint reset complete.", flush=True)
    else:
        print(f"[Core {core_id}] ❌ Checkpoint reset failed.", flush=True)


def main():
    """
    Launch this across all TPU cores.
    Each process corresponds to one core.
    """
    print("🚀 Spawning processes across all TPU cores to reset checkpoints...", flush=True)
    xmp.spawn(run_for_core, args=())


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        # Fallback for non-TPU environments (optional)
        print("⚠️ torch_xla not found. Running single-core fallback.", flush=True)
        reset_checkpoint_samples(0)
