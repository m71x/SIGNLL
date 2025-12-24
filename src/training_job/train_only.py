import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from controller_model import Controller
from training_data_download import training_data_download

# =========================================================================
# TRAINING LOOP
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()

    # -----------------------------
    # Model initialization
    # -----------------------------
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)

    # Sync initial weights
    if rank == 0:
        xm.master_print("Synchronizing initial weights...")
    for p in model.parameters():
        p.data = xm.all_reduce(xm.REDUCE_SUM, p.data) / num_cores
    xm.mark_step()
    xm.rendezvous("weights_synced")

    # =========================================================================
    # STAGE LOOP
    # =========================================================================
    for stage in [1, 2]:

        # -----------------------------
        # Phase setup
        # -----------------------------
        if stage == 1:
            stage_name = "STAGE 1: Backbone + Classifiers (Halting/Gates FROZEN)"
            for p in model.parameters():
                p.requires_grad = True
            for p in model.halting_heads.parameters():
                p.requires_grad = False
            for p in model.entropy_gate_module.parameters():
                p.requires_grad = False
        else:
            stage_name = "STAGE 2: Halting Heads + Gates (Backbone FROZEN)"
            for p in model.parameters():
                p.requires_grad = False
            for p in model.halting_heads.parameters():
                p.requires_grad = True
            for p in model.entropy_gate_module.parameters():
                p.requires_grad = True

        if rank == 0:
            xm.master_print("\n" + "#" * 80)
            xm.master_print(f"STARTING {stage_name}")
            xm.master_print("#" * 80)

        # -----------------------------
        # Optimizer + scheduler
        # -----------------------------
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=flags["lr"], weight_decay=1e-2)

        total_steps = 28 * flags["epochs"] * max(
            1, flags["samples_per_shard"] // flags["batch_size"]
        )
        T_0 = max(1, total_steps // 4)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
        )

        # =========================================================================
        # CHUNK LOOP
        # =========================================================================
        for chunk_idx in range(28):
            filename = f"embeddings_chunk_{chunk_idx}.npz"

            if rank == 0:
                xm.master_print(f"Stage {stage} | Chunk {chunk_idx+1}/28 | Loading {filename}")

            data = training_data_download(
                core_id=rank,
                filename=filename,
                max_entries=flags["samples_per_shard"]
            )

            teacher_cls = torch.from_numpy(data["all_layer_cls_tokens"]).float()
            teacher_lbl = torch.from_numpy(data["classifications"]).long()

            if teacher_cls.shape[1] == 25:
                teacher_cls = teacher_cls[:, 1:25, :]

            # -----------------------------
            # Slice evenly across replicas
            # -----------------------------
            total_bs = flags["batch_size"] * num_cores
            n_groups = teacher_cls.shape[0] // total_bs
            N = n_groups * total_bs

            teacher_cls = teacher_cls[:N]
            teacher_lbl = teacher_lbl[:N]

            dataset = TensorDataset(teacher_cls, teacher_lbl)
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_cores,
                rank=rank,
                shuffle=True,
                drop_last=False
            )

            loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=flags["batch_size"],
                num_workers=0,
                drop_last=True
            )

            # =========================================================================
            # EPOCH LOOP
            # =========================================================================
            for epoch in range(flags["epochs"]):
                sampler.set_epoch(epoch)
                model.train()

                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)

                    halting_logits, class_logits, _ = model(x)

                    # -----------------------------
                    # Stage-specific losses
                    # -----------------------------
                    if stage == 1:
                        # classification only
                        cls_logits = class_logits[:, :, 1]
                        labels = y.float().unsqueeze(1).expand_as(cls_logits)
                        loss = F.binary_cross_entropy_with_logits(cls_logits, labels)
                    else:
                        preds = torch.argmax(class_logits, dim=-1)
                        is_correct = (preds == y.unsqueeze(1)).float()

                        loss_halt = F.binary_cross_entropy_with_logits(
                            halting_logits, is_correct
                        )

                        h = torch.sigmoid(halting_logits).clamp(1e-6, 1 - 1e-6)
                        entropy = -(h * h.log() + (1 - h) * (1 - h).log())

                        loss = loss_halt - 0.0025 * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    xm.mark_step()

                # -----------------------------
                # Flush + logging
                # -----------------------------
                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                xm.mark_step()

                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")

                if rank == 0:
                    lr = scheduler.get_last_lr()[0]
                    xm.master_print("-" * 60)
                    xm.master_print(
                        f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}"
                    )
                    xm.master_print(f"  LR:   {lr:.2e}")
                    xm.master_print(f"  Loss: {loss_sum / num_cores:.4f}")

            xm.rendezvous(f"chunk_end_st{stage}_ch{chunk_idx}")

    # =========================================================================
    # Save
    # =========================================================================
    xm.rendezvous("ready_to_save_final")
    if rank == 0:
        save_path = os.path.expanduser("~/SIGNLL/final_model_stage2_gated.pt")
        torch.save(model.state_dict(), save_path)
        xm.master_print("✅ Training complete.")
    xm.rendezvous("save_complete_safe_exit")


def _mp_fn(rank, flags):
    train_loop(rank, flags)


if __name__ == "__main__":
    FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 5,
        "samples_per_shard": 39000
    }

    print("Starting Stage-1 + Stage-2 Training (TPU-safe).")
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method="fork")
