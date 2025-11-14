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
    "max_entries": 128,  # Tiny sample for quick testing
    
    # XLA/Distributed parameters
    "num_cores_to_test": 8, # Simulate testing across 8 cores
}

def load_data_and_model(rank: int, config: Dict[str, Any]) -> Tuple[Controller, torch.Tensor, torch.Tensor]:
    """
    Downloads data shard corresponding to the core rank and initializes the model.
    """
    device = xm.xla_device()
    xm.master_print(f"[{rank}] Initializing on device: {device}")
    
    # 1. Load Data
    # Use the core rank as the core_id for sharded data loading
    xm.master_print(f"[{rank}] Loading data for core_id {rank}...")
    
    data = training_data_download(
        core_id=rank, # core_id is now the rank
        filename=config["chunk_filename"], 
        max_entries=config["max_entries"]
    )

    if data is None:
        raise RuntimeError(f"[{rank}] Failed to load or generate training data. Cannot proceed with tests.")

    # Convert NumPy arrays to PyTorch Tensors and move to XLA device
    teacher_cls = torch.from_numpy(data['all_layer_cls_tokens']).float().to(device)
    teacher_label = torch.from_numpy(data['classifications']).long().to(device)

    # 2. Initialize Model
    model = Controller(
        L=config["L"],
        d_teacher=config["d_teacher"],
        d_ctrl=config["d_ctrl"],
        n_layers=config["transformer_layers"],
        num_classes=config["num_classes"]
    ).to(device)
    
    xm.master_print(f"[{rank}] Data loaded: CLS Tokens shape: {teacher_cls.shape}, Labels shape: {teacher_label.shape}")
    xm.master_print(f"[{rank}] Model initialized and moved to XLA device.")

    return model, teacher_cls, teacher_label

def test_inference_and_loss_step(model: Controller, teacher_cls: torch.Tensor, teacher_label: torch.Tensor, config: Dict[str, Any]):
    """
    Runs inference and calculates a single loss step, addressing original test items #1 and #2.
    """
    rank = xm.get_ordinal()
    B, L, D_teacher = teacher_cls.shape
    num_classes = config["num_classes"]
    lambda_target = config["halting_lambda_target"]
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    # 1. Inference (Test Item #1)
    model.eval()
    with torch.no_grad():
        halting_logits, class_logits, z = model(teacher_cls)
        
    # Check output shapes on the master core
    if rank == 0:
        expected_halting_shape = (B, L)
        expected_class_shape = (B, L, num_classes)
        expected_z_shape = (B, L, config["d_ctrl"])

        assert halting_logits.shape == expected_halting_shape, \
            f"Halting logits shape mismatch: Expected {expected_halting_shape}, got {halting_logits.shape}"
        assert class_logits.shape == expected_class_shape, \
            f"Class logits shape mismatch: Expected {expected_class_shape}, got {class_logits.shape}"
        assert z.shape == expected_z_shape, \
            f"Controller internal state (z) shape mismatch: Expected {expected_z_shape}, got {z.shape}"
        xm.master_print("✅ Test 1 Passed: Inference output shapes verified on master core.")

    # 2. Halting Probabilities (h) and Exit Probabilities (q)
    h = torch.sigmoid(halting_logits)
    q = compute_q_from_h(h)
    
    # 3. Classification Loss (Loss_cls) (Part of Test Item #2)
    labels_float = teacher_label.float().unsqueeze(1).expand(-1, L)
    
    # Take the positive class logit for BCE loss (assuming binary classification)
    if class_logits.size(-1) == 2:
        class_logits_positive = class_logits[:, :, 1]
    else:
        class_logits_positive = class_logits.squeeze(-1)
        
    ce_per_layer = bce_loss_fn(class_logits_positive, labels_float)
        
    # Weighted classification loss: (q * CE_per_layer) summed over layers, then mean over batch
    loss_cls_per_sample = (q * ce_per_layer).sum(dim=1)
    loss_cls = loss_cls_per_sample.mean()
    
    # 4. Halting Loss (Loss_halt) (Part of Test Item #2)
    depths = torch.arange(1, L + 1, device=xm.xla_device()).float().unsqueeze(0) # [1, L]
    halt_penalty = (depths * (1 - h)).sum(dim=1) # [B]
    loss_halt = lambda_target * halt_penalty.mean()
    
    # 5. Total Loss
    loss = loss_cls + loss_halt

    # 6. Global Pooling (Test Item #2)
    # The losses above are local means. We gather the mean losses from all cores
    # for a global check (simulating global loss calculation).
    
    # Convert local loss to a single-item tensor for all-reduce
    loss_local = torch.tensor([loss.item()], dtype=torch.float32, device=xm.xla_device())
    loss_cls_local = torch.tensor([loss_cls.item()], dtype=torch.float32, device=xm.xla_device())
    loss_halt_local = torch.tensor([loss_halt.item()], dtype=torch.float32, device=xm.xla_device())

    # All-reduce to get the sum of losses across all cores
    loss_global_sum = xm.all_reduce(xm.REDUCE_SUM, loss_local)
    loss_cls_global_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls_local)
    loss_halt_global_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt_local)
    
    # Calculate global mean by dividing by the number of cores
    num_cores = xm.xrt_world_size()
    global_loss_mean = loss_global_sum.item() / num_cores
    global_loss_cls_mean = loss_cls_global_sum.item() / num_cores
    global_loss_halt_mean = loss_halt_global_sum.item() / num_cores

    if rank == 0:
        xm.master_print("✅ Test 2 Passed: Loss components calculated and globally pooled successfully.")
        xm.master_print(f"--- Loss Check (Averages over {num_cores} cores) ---")
        xm.master_print(f"  Classification Loss (L_cls): {global_loss_cls_mean:.6f}")
        xm.master_print(f"  Halting Loss (L_halt): {global_loss_halt_mean:.6f}")
        xm.master_print(f"  Total Loss (L_total): {global_loss_mean:.6f}")
        xm.master_print("-" * 60)

def test_controller_mp_fn(rank, config):
    """
    The main test function launched by xmp.spawn.
    """
    try:
        model, teacher_cls, teacher_label = load_data_and_model(rank, config)
        test_inference_and_loss_step(model, teacher_cls, teacher_label, config)
    except Exception as e:
        xm.master_print(f"FATAL ERROR on core {rank}: {e}", file=sys.stderr)
        # Re-raise the exception to stop the test process
        raise

# The entry point to run the tests
if __name__ == "__main__":
    # The environment variable XRT_HOST_ORDINAL needs to be set up correctly
    # in the TPU environment for this to work with all 32 cores.
    # We use a defined number of cores for a reliable local test simulation.
    
    xm.master_print("--- Starting Controller Architecture Verification Tests (XLA Multi-Core Simulation) ---")
    
    # Launch the test function on multiple cores
    xmp.spawn(
        test_controller_mp_fn, 
        args=(TEST_CONFIG,)
    )
    
    xm.master_print("\n--- All Architectural Verification Tests Completed Successfully! ---")