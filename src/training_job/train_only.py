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

    diag_sample_pos = None
    diag_sample_neg = None

    # =========================================================================
    # STAGE LOOP (Skips Stage 1 for testing Stage 2 stability)
    # =========================================================================
    for stage in [2]:

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

        # --- SAFE SCHEDULER LOGIC ---
        SAFE_SHARD_SIZE = 38000 
        batches_per_chunk = SAFE_SHARD_SIZE // flags["batch_size"]
        total_steps_in_stage = 28 * flags["epochs"] * batches_per_chunk
        T_0 = max(1, total_steps_in_stage // 4)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0, 
            T_mult=1,
            eta_min=1e-6
        )
        
        if rank == 0:
            xm.master_print(f"Scheduler T_0: {T_0} steps (Total steps: {total_steps_in_stage})")

        # =========================================================================
        # CHUNK LOOP
        # =========================================================================
        for chunk_idx in range(28):
            filename = f"embeddings_chunk_{chunk_idx}.npz"

            if rank == 0:
                xm.master_print(
                    f"Stage {stage} | Chunk {chunk_idx+1}/28 | Loading {filename}"
                )

            data = training_data_download(
                core_id=rank,
                filename=filename,
                max_entries=flags["samples_per_shard"]
            )

            teacher_cls = torch.from_numpy(data["all_layer_cls_tokens"]).float()
            teacher_lbl = torch.from_numpy(data["classifications"]).long()

            if teacher_cls.shape[1] == 25:
                teacher_cls = teacher_cls[:, 1:25, :]

            # Truncate for safety
            if teacher_cls.shape[0] >= SAFE_SHARD_SIZE:
                teacher_cls = teacher_cls[:SAFE_SHARD_SIZE]
                teacher_lbl = teacher_lbl[:SAFE_SHARD_SIZE]
            
            dataset = TensorDataset(teacher_cls, teacher_lbl)
            sampler = torch.utils.data.RandomSampler(dataset)

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
                model.train()
                diag_sample_pos = None
                diag_sample_neg = None

                for batch_idx, (x, y) in enumerate(loader):
                    x = x.to(device)
                    y = y.to(device)

                    halting_logits, class_logits, z = model(x)

                    # -----------------------------
                    # Stage-specific losses
                    # -----------------------------
                    if stage == 1:
                        # (Stage 1 logic is skipped in this script)
                        pass

                    else: # STAGE 2
                        preds = torch.argmax(class_logits, dim=-1)
                        
                        # --- FIX 1: TENSOR SHAPE [B, L, 1] ---
                        is_correct = (preds == y.unsqueeze(1)).float().unsqueeze(-1)

                        # 1. Class-Aware Rebalancing Weights
                        n_pos = (y == 1).sum().float()
                        n_neg = (y == 0).sum().float()
                        neg_weight_val = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)

                        # --- FIX 2: STATIC GRAPH + CORRECT BROADCASTING ---
                        # Use torch.where for static graph
                        weights = torch.where(
                            y == 0, 
                            neg_weight_val, 
                            torch.tensor(1.0, device=device)
                        )
                        # CRITICAL FIX: Reshape to [Batch, 1, 1] to broadcast against [Batch, Length, 1]
                        sample_weights = weights.view(-1, 1, 1)

                        # 2. Halting Loss
                        loss_halt = F.binary_cross_entropy_with_logits(
                            halting_logits, 
                            is_correct, 
                            weight=sample_weights
                        )

                        # 3. Entropy Regularization
                        h = torch.sigmoid(halting_logits)
                        h_safe = h.clamp(min=1e-6, max=1.0 - 1e-6)
                        entropy_weight = 0.0025
                        h_entropy = -(h_safe * h_safe.log() + (1 - h_safe) * (1 - h_safe).log())
                        loss_entropy = -entropy_weight * h_entropy.mean()

                        loss = loss_halt + loss_entropy

                        if rank == 0 and batch_idx == 0:
                            with torch.no_grad():
                                def extract_sample(label_val):
                                    indices = (y == label_val).nonzero(as_tuple=True)[0]
                                    if indices.numel() > 0:
                                        idx = indices[0]
                                        return {
                                            "halt": halting_logits[idx].detach().cpu(),
                                            "lbl": y[idx].detach().cpu()
                                        }
                                    return None

                                diag_sample_pos = extract_sample(1)
                                diag_sample_neg = extract_sample(0)

                    optimizer.zero_grad()
                    loss.backward()
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    xm.mark_step()

                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                xm.mark_step()
                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")

                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    xm.master_print("-" * 60)
                    xm.master_print(
                        f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}"
                    )
                    xm.master_print(f"  LR:         {current_lr:.2e}")
                    xm.master_print(f"  Total Loss: {loss_sum / num_cores:.4f}")
                    if stage == 2:
                        xm.master_print(f"  Halt Loss:  {loss_sum / num_cores:.4f}")

                    def format_sample(data, name):
                        if data is None: return f"  {name}: No sample found."
                        out = [f"  > {name} (Label {data['lbl'].item()}):"]
                        h_probs = torch.sigmoid(data["halt"])
                        if h_probs.dim() > 1: h_probs = h_probs.squeeze(-1)
                        out.append(f"    HALT Probs: {[f'{p:.2f}' for p in h_probs.tolist()]}")
                        return "\n".join(out)

                    xm.master_print("  DIAGNOSTICS:")
                    xm.master_print(format_sample(diag_sample_pos, "Sample POS"))
                    xm.master_print(format_sample(diag_sample_neg, "Sample NEG"))

            xm.rendezvous(f"chunk_end_st{stage}_ch{chunk_idx}")

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
        "d_ctrl": 1024,
        "transformer_layers": 4,
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 5,
        "samples_per_shard": 39000
    }

    print("Starting Stage-2 Training (TPU-safe, Vectorized).")
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method="fork")