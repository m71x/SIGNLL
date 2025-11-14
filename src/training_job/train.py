#actual training script

# controller_train_xla.py
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from controller_model import Controller, compute_q_from_h
from controller_utils import run_teacher_and_get_cls, inference_once

#only use up to chunk 28, file sizes vary acrss cores for chunks 29-31
def train_loop(rank, flags):
    device = xm.torch_xla.device()
    print(f"[Core {rank}] Using device:", device)

    # Unpack flags
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    num_epochs = flags["epochs"]
    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
    batch_size = flags["batch_size"]

    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        for step in range(100):  # placeholder
            global_step += 1
            batch_inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            #switch out for data loaded from training_data_download.py, since this is bigger, probably shoudl put it outside of this for loop
            teacher_cls, teacher_label = run_teacher_and_get_cls(batch_inputs, device)

            halting_logits, class_logits, _ = model(teacher_cls)
            h = torch.sigmoid(halting_logits)
            q = compute_q_from_h(h)

            labels = teacher_label.float().unsqueeze(1).expand(-1, L)
            ce_per_layer = bce_loss_fn(class_logits, labels)
            loss_cls = (q * ce_per_layer).sum(dim=1).mean()

            depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
            halt_penalty = (depths * (1 - h)).sum(dim=1)
            progress = min(1.0, epoch / max(1.0, num_epochs - 1))
            lambda_now = lambda_start + (lambda_target - lambda_start) * progress
            loss_halt = lambda_now * halt_penalty.mean()

            loss = loss_cls + loss_halt
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)

            if global_step % 10 == 0:
                xm.master_print(
                    f"Step {global_step} | loss={loss.item():.4f} cls={loss_cls.item():.4f} halt={loss_halt.item():.4f} mean_h={h.mean().item():.4f}"
                )

        xm.master_print(f"Epoch {epoch} done. Loss={loss.item():.4f}")

    xm.master_print("Training finished. Elapsed:", time.time() - start_time)


def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    train_loop(rank, flags)


if __name__ == "__main__":
    FLAGS = {
        "d_ctrl": 256,
        "transformer_layers": 4,
        "lr": 3e-4,
        "batch_size": 128,
        "lambda_start": 0.0,
        "lambda_target": 0.01,
        "alpha": 0.9,
        "epochs": 5,
    }

    nprocs = xm.xrt_world_size() if 'XRT_TPU_CONFIG' in os.environ else 1
    #should get rid of nprocs
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=nprocs, start_method='fork')
