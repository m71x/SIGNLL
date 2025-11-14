import numpy as np
import torch
import torch.nn as nn
import time
import sys
from typing import Dict, Any, Tuple

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

# --- Test Configuration ---
TEST_CONFIG = {
    "L": 24,
    "d_teacher": 1024,
    "d_ctrl": 256,
    "transformer_layers": 4,
    "halting_lambda_target": 0.01,
    "num_classes": 2,
    "chunk_filename": "embeddings_chunk_0.npz",
    "max_entries": 1,
    "num_cores_to_test": 32,
}

# ================================================================
# LOADING MODEL + DATA
# ================================================================
def load_data_and_model(rank: int, config: Dict[str, Any]):
    device = xm.torch_xla.device()

    xm.master_print(f"[{rank}] Initializing on device {device}")

    data = training_data_download(
        core_id=rank,
        filename=config["chunk_filename"],
        max_entries=config["max_entries"]
    )

    if data is None:
        raise RuntimeError(f"[{rank}] Failed to load data")

    teacher_cls = torch.from_numpy(data['all_layer_cls_tokens']).float().to(device)
    teacher_label = torch.from_numpy(data['classifications']).long().to(device)

    xm.master_print(f"[{rank}] Data loaded: teacher_cls {teacher_cls.shape}, labels {teacher_label.shape}")

    model = Controller(
        L=config["L"],
        d_teacher=config["d_teacher"],
        d_ctrl=config["d_ctrl"],
        n_layers=config["transformer_layers"],
        num_classes=config["num_classes"]
    ).to(device)

    xm.master_print(f"[{rank}] Model initialized.")

    return model, teacher_cls, teacher_label

# ================================================================
# TEST 1 — INFERENCE + LOSS + PRINT LOCAL + GLOBAL LOSS
# ================================================================
def test_inference_and_loss_step(model, teacher_cls, teacher_label, config):

    rank = xm.get_ordinal()

    B, L, D = teacher_cls.shape
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    lambda_target = config["halting_lambda_target"]

    # ---------- Inference ----------
    model.eval()
    with torch.no_grad():
        halting_logits, class_logits, z = model(teacher_cls)

    # ---------- Build losses ----------
    h = torch.sigmoid(halting_logits)
    q = compute_q_from_h(h)

    labels_float = teacher_label.float().unsqueeze(1).expand(-1, L)
    class_logits_positive = class_logits[:, :, 1]

    ce = bce_loss_fn(class_logits_positive, labels_float)
    loss_cls = (q * ce).sum(dim=1).mean()

    depths = torch.arange(1, L + 1, device=teacher_cls.device).unsqueeze(0)
    loss_halt = lambda_target * ((depths * (1 - h)).sum(dim=1).mean())

    loss = loss_cls + loss_halt

    # ---------- Print per-core values ----------
    xm.master_print(f"[core {rank}] loss_local      = {loss.item():.6f}")
    xm.master_print(f"[core {rank}] loss_cls_local  = {loss_cls.item():.6f}")
    xm.master_print(f"[core {rank}] loss_halt_local = {loss_halt.item():.6f}")

    # ---------- All-reduce for global pooling ----------
    l_local = torch.tensor([loss.item()], device=loss.device)
    lc_local = torch.tensor([loss_cls.item()], device=loss.device)
    lh_local = torch.tensor([loss_halt.item()], device=loss.device)

    l_sum = xm.all_reduce(xm.REDUCE_SUM, l_local)
    lc_sum = xm.all_reduce(xm.REDUCE_SUM, lc_local)
    lh_sum = xm.all_reduce(xm.REDUCE_SUM, lh_local)

    N = xm.xrt_world_size()

    if rank == 0:
        xm.master_print("----- Global Loss Pooling -----")
        xm.master_print(f"Global L_cls   = {lc_sum.item()/N:.6f}")
        xm.master_print(f"Global L_halt  = {lh_sum.item()/N:.6f}")
        xm.master_print(f"Global L_total = {l_sum.item()/N:.6f}")
        xm.master_print("--------------------------------")


# ================================================================
# TEST 2 — UPDATE STEP (with ALL-REDUCE and checkpoint save)
# ================================================================
def test_update_step(model, teacher_cls, teacher_label, config):

    rank = xm.get_ordinal()
    device = xm.torch_xla.device()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    lambda_target = config["halting_lambda_target"]

    # ---------- Forward ----------
    model.train()
    halting_logits, class_logits, z = model(teacher_cls)

    h = torch.sigmoid(halting_logits)
    q = compute_q_from_h(h)

    B, L, _ = teacher_cls.shape
    labels = teacher_label.float().unsqueeze(1).expand(-1, L)
    class_logits_positive = class_logits[:, :, 1]

    ce = bce_loss_fn(class_logits_positive, labels)
    loss_cls = (q * ce).sum(dim=1).mean()

    depths = torch.arange(1, L + 1, device=device).unsqueeze(0)
    loss_halt = lambda_target * ((depths * (1 - h)).sum(dim=1).mean())

    loss = loss_cls + loss_halt

    xm.master_print(f"[core {rank}] BEFORE update: loss = {loss.item():.6f}")

    # ---------- All-reduce BEFORE update ----------
    l_local = torch.tensor([loss.item()], device=device)
    l_sum = xm.all_reduce(xm.REDUCE_SUM, l_local)
    if rank == 0:
        xm.master_print(f"[core 0] BEFORE update global_loss = {l_sum.item()/xm.xrt_world_size():.6f}")

    # ---------- Backprop ----------
    optimizer.zero_grad()
    loss.backward()

    # ---------- Optimizer step (sync gradients) ----------
    xm.optimizer_step(optimizer)

    # ---------- Recompute loss AFTER update ----------
    with torch.no_grad():
        halting_logits2, class_logits2, _ = model(teacher_cls)

    h2 = torch.sigmoid(halting_logits2)
    q2 = compute_q_from_h(h2)

    ce2 = bce_loss_fn(class_logits2[:, :, 1], labels)
    loss_after = (q2 * ce2).sum(dim=1).mean()

    xm.master_print(f"[core {rank}] AFTER update: loss = {loss_after.item():.6f}")

    # ---------- All-reduce AFTER update ----------
    l2_local = torch.tensor([loss_after.item()], device=device)
    l2_sum = xm.all_reduce(xm.REDUCE_SUM, l2_local)

    if rank == 0:
        xm.master_print(f"[core 0] AFTER update global_loss = {l2_sum.item()/xm.xrt_world_size():.6f}")

        # Save checkpoint ONCE
        xm.master_print("[core 0] Saving checkpoint (placeholder path)")
        # torch.save(model.state_dict(), "/tmp/controller_checkpoint.pt")


# ================================================================
# MAIN MULTIPROCESS FUNCTION
# ================================================================
def test_controller_mp_fn(rank, config):

    try:
        model, teacher_cls, teacher_label = load_data_and_model(rank, config)

        # reduce CLS from 25 to 24 layers
        teacher_cls = teacher_cls[:, 1:25, :]

        # Test 1: inference + loss + global pooling
        test_inference_and_loss_step(model, teacher_cls, teacher_label, config)

        # Test 2: update step (loss before/after + all-reduce)
        test_update_step(model, teacher_cls, teacher_label, config)

    except Exception as e:
        xm.master_print(f"[FATAL ERROR on core {rank}] {e}")
        raise


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":

    xmp.spawn(
        test_controller_mp_fn,
        args=(TEST_CONFIG,)
    )

    xm.master_print("\n--- All Tests Completed Successfully ---")
