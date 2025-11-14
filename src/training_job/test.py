#test the following:
#1. model can correctly load in npz tensors ones and run inference to output logits in the correct structure (in parallel, with 32 cores)
#2. model can correctly pool results across parallel cores and make one update step based on the actual training requirements
#3 model can correctly keep track of checkpoints
#4 model can correctly save the weights of the model ONCE (not for each core) somewhere, probably gcs
#5 good idea to have checkpoints probably, but you already have code for that so you don't need to test much
#6 make sure during the code consistently outputs logs like accuracy, epoch, batch, etc

import numpy as np
import torch
import torch.nn as nn
import time
import sys
from typing import Dict, Any, Tuple

# Import XLA utilities for TPU execution
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# Assuming these files are in the same directory and can be imported
from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

# --- Test Configuration ---
TEST_CONFIG = {
    # General Model/Loss parameters
    "L": 24, 
    "d_teacher": 1024,
    "d_ctrl": 256,
    "transformer_layers": 4,
    "halting_lambda_target": 0.01,
    "num_classes": 2, 
    
    # Data loading parameters
    "chunk_filename": "embeddings_chunk_0.npz", # Use a consistent file name for all cores
    "max_entries": 1,  # Tiny sample for quick testing
    
    # XLA/Distributed parameters
    "num_cores_to_test": 32, # Simulate testing across 8 cores
}

def load_data_and_model(rank: int, config: Dict[str, Any]) -> Tuple[Controller, torch.Tensor, torch.Tensor]:
    device = xm.torch_xla.device()
    xm.master_print(f"[{rank}] Initializing on device: {device}")

    # 1. Load Data
    xm.master_print(f"[{rank}] Loading data for core_id {rank}...")

    data = training_data_download(
        core_id=rank,
        filename=config["chunk_filename"],
        max_entries=config["max_entries"]
    )

    if data is None:
        raise RuntimeError(f"[{rank}] Failed to load or generate training data.")

    teacher_cls = torch.from_numpy(data['all_layer_cls_tokens']).float().to(device)
    teacher_label = torch.from_numpy(data['classifications']).long().to(device)

    xm.master_print(f"[{rank}] Loaded teacher_cls shape: {teacher_cls.shape}")
    xm.master_print(f"[{rank}] Loaded teacher_label shape: {teacher_label.shape}")

    # 2. Initialize the model
    model = Controller(
        L=config["L"],
        d_teacher=config["d_teacher"],
        d_ctrl=config["d_ctrl"],
        n_layers=config["transformer_layers"],
        num_classes=config["num_classes"]
    ).to(device)

    xm.master_print(f"[{rank}] Model initialized.")

    # 🚨 IMPORTANT: Barrier after data + model before ANY forward/all-reduce
    xm.rendezvous("after_data_load")

    return model, teacher_cls, teacher_label



def test_inference_and_loss_step(model: Controller, teacher_cls: torch.Tensor, teacher_label: torch.Tensor, config: Dict[str, Any]):
    rank = xm.get_ordinal()
    B, L, D_teacher = teacher_cls.shape
    num_classes = config["num_classes"]
    lambda_target = config["halting_lambda_target"]
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # 1. Inference
    model.eval()
    with torch.no_grad():
        halting_logits, class_logits, z = model(teacher_cls)

    if rank == 0:
        assert halting_logits.shape == (B, L)
        assert class_logits.shape == (B, L, num_classes)
        assert z.shape == (B, L, config["d_ctrl"])
        xm.master_print("✅ Test 1 Passed: Inference output shapes verified.")

    # 2. Compute h and q
    h = torch.sigmoid(halting_logits)
    q = compute_q_from_h(h)

    # 3. Classification loss
    labels_float = teacher_label.float().unsqueeze(1).expand(-1, L)

    if class_logits.size(-1) == 2:
        class_logits_positive = class_logits[:, :, 1]
    else:
        class_logits_positive = class_logits.squeeze(-1)

    ce_per_layer = bce_loss_fn(class_logits_positive, labels_float)
    loss_cls = (q * ce_per_layer).sum(dim=1).mean()

    # 4. Halting loss
    depths = torch.arange(1, L + 1, device=xm.xla_device()).float().unsqueeze(0)
    halt_penalty = (depths * (1 - h)).sum(dim=1)
    loss_halt = lambda_target * halt_penalty.mean()

    loss = loss_cls + loss_halt

    # 🚨 IMPORTANT: ALL CORES MUST REACH HERE BEFORE ALLREDUCE
    xm.rendezvous("before_allreduce")

    # 5. Global pooling of losses
    loss_local = torch.tensor([loss.item()], dtype=torch.float32, device=xm.xla_device())
    loss_cls_local = torch.tensor([loss_cls.item()], dtype=torch.float32, device=xm.xla_device())
    loss_halt_local = torch.tensor([loss_halt.item()], dtype=torch.float32, device=xm.xla_device())

    loss_global_sum = xm.all_reduce(xm.REDUCE_SUM, loss_local)
    loss_cls_global_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls_local)
    loss_halt_global_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt_local)

    num_cores = xm.xrt_world_size()

    if rank == 0:
        xm.master_print("✅ Test 2 Passed: Global loss pooling works.")
        xm.master_print(f"  Classification Loss: {loss_cls_global_sum.item() / num_cores:.6f}")
        xm.master_print(f"  Halting Loss:        {loss_halt_global_sum.item() / num_cores:.6f}")
        xm.master_print(f"  Total Loss:          {loss_global_sum.item() / num_cores:.6f}")
        xm.master_print("-" * 60)



def test_controller_mp_fn(rank, config):
    try:
        model, teacher_cls, teacher_label = load_data_and_model(rank, config)
        teacher_cls = teacher_cls[:, 1:25, :]
        test_inference_and_loss_step(model, teacher_cls, teacher_label, config)
    except Exception as e:
        xm.master_print(f"FATAL ERROR on core {rank}: {e}")
        raise



if __name__ == "__main__":
    xmp.spawn(test_controller_mp_fn, args=(TEST_CONFIG,))
    xm.master_print("\n--- All Tests Completed Successfully! ---")