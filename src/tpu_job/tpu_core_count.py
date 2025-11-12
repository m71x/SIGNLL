import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def main_worker(index):
    core_id = xm.get_ordinal()            # global core ID (0–31)
    local_core_id = xm.get_local_ordinal() # core ID within the worker (0–3)
    world_size = xm.xrt_world_size()      # total number of TPU cores (should be 32)
    print(f"[Worker TEST] Global Core: {core_id:02d}, Local Core: {local_core_id}, World Size: {world_size}", flush=True)

if __name__ == "__main__":
    print("Launching TPU core test...")
    xmp.spawn(main_worker, args=(), nprocs=None)
