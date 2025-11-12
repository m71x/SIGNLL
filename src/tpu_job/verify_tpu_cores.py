# check_tpu_world.py

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import os

def check_world_size(_):
    # This function is run on every available TPU core
    core_id = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    print(f"[{core_id}] World Size: {world_size}", flush=True)

if __name__ == '__main__':
    # Use the same spawning mechanism as your main.py
    print("--- Attempting to spawn workers to check TPU world size ---", flush=True)
    # nprocs=xm.xrt_world_size() is implicitly used if nprocs is not set
    xmp.spawn(check_world_size, args=(), nprocs=None)
    print("--- Finished world size check ---", flush=True)