import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    device = xm.torch_xla.device()
    print(f"[{index}] Device: {device}")
    xm.rendezvous("init")  # forces all workers to sync once
    print(f"[{index}] Sync successful")

if __name__ == "__main__":
    xmp.spawn(_mp_fn, nprocs=8)


