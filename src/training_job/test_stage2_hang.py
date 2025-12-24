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

    # 1. Model Initialization
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
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()

    xm.rendezvous("weights_synced")

    # Calculate step counts
    total_samples = flags["samples_per_shard"]
    # NOTE: num_batches_per_chunk will be recomputed per-chunk after slicing.
    global_step = 0
    start_time = time.time()

    diag_sample_pos = None
    diag_sample_neg = None

    # =========================================================================
    # STAGE LOOP: Skip stage 1, only run stage 2 (as requested)
    # =========================================================================
    for stage in [2]:

        # --- PHASE SETUP ---
        if stage == 1:
            stage_name = "STAGE 1: Backbone & Classifiers (Halting & Gates FROZEN)"
            for param in model.parameters():
                param.requires_grad = True

            # Freeze halting heads
            for param in model.halting_heads.parameters():
                param.requires_grad = False

            # Freeze optimized entropy gate module
            for param in model.entropy_gate_module.parameters():
                param.requires_grad = False

        elif stage == 2:
            stage_name = "STAGE 2: Halting Heads + Entropy Gates (Backbone FROZEN)"
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze halting heads
            for param in model.halting_heads.parameters():
                param.requires_grad = True

            # Unfreeze optimized entropy gate module
            for param in model.entropy_gate_module.parameters():
                param.requires_grad = True

        if rank == 0:
            xm.master_print(f"\n{'#'*80}")
            xm.master_print(f"STARTING {stage_name}")
            xm.master_print(f"{'#'*80}")

        # --- OPTIMIZER SETUP ---
        model_params = [p for p in model.parameters() if p.requires_grad]

        if rank == 0:
            num_params = sum(p.numel() for p in model_params)
            xm.master_print(f"Trainable parameters: {num_params:,}")

        optimizer = optim.AdamW(model_params, lr=flags["lr"], weight_decay=1e-2)

        # --- SCHEDULER SETUP ---
        # We'll compute total_steps_in_stage using the per-chunk batch count after slicing
        # For now keep the same T_0 calc base — we'll set scheduler once per stage below.
        # We'll recompute scheduler if needed once we know num_batches_per_chunk. For simplicity,
        # keep the same heuristic you had, it's conservative.
        total_steps_in_stage = 28 * flags["epochs"] * max(1, (flags["samples_per_shard"] // flags["batch_size"]))
        T_0 = max(1, total_steps_in_stage // 4)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=1e-6
        )
        if rank == 0:
            xm.master_print(f"Scheduler: CosineAnnealingWarmRestarts, T_0={T_0} steps")

        # Loop through chunks
        for chunk_idx in range(28):
            current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"

            if rank == 0:
                xm.master_print(f"Stage {stage} | Chunk {chunk_idx + 1}/28 | Loading {current_chunk_filename}")

            # --- Load Data ---
            data = training_data_download(
                core_id=rank,
                filename=current_chunk_filename,
                max_entries=flags["samples_per_shard"]
            )

            if data is None:
                raise RuntimeError(f"[Core {rank}] Failed load chunk {chunk_idx}")

            teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
            teacher_label_full = torch.from_numpy(data['classifications']).long()

            # --- Soft Targets ---
            if 'teacher_logits' in data:
                t_logits = torch.from_numpy(data['teacher_logits']).float()
                T_distill = 2.0
                teacher_log_probs_full = F.log_softmax(t_logits / T_distill, dim=-1)
            else:
                num_classes = 2
                smoothing = 0.1
                t_one_hot = torch.zeros(teacher_label_full.size(0), num_classes).scatter_(
                    1, teacher_label_full.unsqueeze(1), 1
                )
                teacher_probs_full = t_one_hot * (1.0 - smoothing) + (smoothing / num_classes)
                teacher_log_probs_full = torch.log(teacher_probs_full.clamp(min=1e-10))

            if teacher_cls_full.shape[1] == 25:
                teacher_cls_full = teacher_cls_full[:, 1:25, :]

            # --- Data Slicing: make total samples divisible by (batch_size * num_cores)
            N_total_local = teacher_cls_full.shape[0]
            total_batch_size = flags["batch_size"] * num_cores
            n_full_groups = N_total_local // total_batch_size  # number of full cross-core groups
            if n_full_groups == 0:
                raise RuntimeError(f"[Core {rank}] Chunk {chunk_idx} has too few samples ({N_total_local}) for total_batch_size={total_batch_size}")

            N_target = n_full_groups * total_batch_size
            # Slice all arrays to N_target
            teacher_cls_full = teacher_cls_full[:N_target]
            teacher_label_full = teacher_label_full[:N_target]
            teacher_log_probs_full = teacher_log_probs_full[:N_target]

            # Now each replica will get exactly N_target / num_cores samples
            per_replica_samples = N_target // num_cores
            num_batches_per_chunk = per_replica_samples // flags["batch_size"]

            # Safety: assert integer division
            assert per_replica_samples % flags["batch_size"] == 0, (
                f"per_replica_samples ({per_replica_samples}) must be divisible by batch_size ({flags['batch_size']})"
            )

            # Class Weighting
            neg_samples = (teacher_label_full == 0).sum().item()
            pos_samples = (teacher_label_full == 1).sum().item()
            pos_weight_val = neg_samples / (pos_samples + 1e-6)
            pos_weight_tensor = torch.tensor([pos_weight_val]).float().to(device)

            bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor).to(device)

            dataset = TensorDataset(teacher_cls_full, teacher_label_full, teacher_log_probs_full)

            # -----------------------------
            # Use DistributedSampler so each replica sees a disjoint slice but identical counts
            # -----------------------------
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_cores,
                rank=rank,
                shuffle=True,
                drop_last=False  # we already sliced to be evenly divisible across replicas
            )

            # Use num_workers=0 for TPU VM stability
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=flags["batch_size"],
                drop_last=True,
                num_workers=0,
                pin_memory=False
            )

            # --- Sanity check (collective) for batch counts ---
            local_num_batches = torch.tensor([len(data_loader)], device=device)
            summed = xm.all_reduce(xm.REDUCE_SUM, local_num_batches)
            avg = (summed / num_cores).item()
            # If not equal, averaged value != local -> print diagnostic (collective, so safe)
            if local_num_batches.item() != avg:
                # This should never happen given slicing above. If it does, print helpful info.
                xm.master_print(f"[Rank {rank}] WARNING: batch_count mismatch: local={local_num_batches.item()}, avg={avg}")

            # --- Epoch Loop ---
            for epoch in range(flags["epochs"]):
                # Important: set epoch so DistributedSampler shuffling is consistent across replicas
                sampler.set_epoch(epoch)
                model.train()
                diag_sample_pos = None
                diag_sample_neg = None

                for batch_idx, (teacher_cls, teacher_label, teacher_log_probs) in enumerate(data_loader):
                    if stage == 2:
                        global_step += 1

                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    teacher_log_probs = teacher_log_probs.to(device)

                    # --- STAGE 2: Standard Training (Halting + Gates) ---
                    # Removed SAM Closure, using standard forward/backward
                    halting_logits, class_logits, z = model(teacher_cls)

                    predictions = torch.argmax(class_logits, dim=-1)
                    is_correct = (predictions == teacher_label.unsqueeze(1)).float()

                    # CLASS-AWARE REBALANCING
                    n_pos = (teacher_label == 1).sum().float()
                    n_neg = (teacher_label == 0).sum().float()
                    neg_weight_val = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)

                    sample_weights = torch.ones_like(halting_logits)
                    # broadcast teacher_label to match halting_logits shape: [B, L]
                    # halting_logits shape: [B, L]
                    # teacher_label shape: [B]
                    sample_weights[teacher_label == 0] = neg_weight_val.item()

                    # Use BCEWithLogitsLoss for stability
                    loss_halt = F.binary_cross_entropy_with_logits(
                        halting_logits,
                        is_correct,
                        weight=sample_weights
                    )

                    # Entropy regularization
                    h = torch.sigmoid(halting_logits)
                    h_safe = h.clamp(min=1e-6, max=1.0 - 1e-6)

                    entropy_weight = 0.0025
                    h_entropy = -(h_safe * h_safe.log() + (1 - h_safe) * (1 - h_safe).log())
                    loss_entropy = -entropy_weight * h_entropy.mean()

                    loss = loss_halt + loss_entropy

                    if rank == 0 and batch_idx == 0:
                        with torch.no_grad():
                            def extract_sample(label_val):
                                indices = (teacher_label == label_val).nonzero(as_tuple=True)[0]
                                if indices.numel() > 0:
                                    idx = indices[0]
                                    return {
                                        'cls': class_logits[idx].detach().cpu(),
                                        'halt': halting_logits[idx].detach().cpu(),
                                        'lbl': teacher_label[idx].detach().cpu()
                                    }
                                return None

                            diag_sample_pos = extract_sample(1)
                            diag_sample_neg = extract_sample(0)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    xm.mark_step()

                # end of epoch for this chunk
                # synchronize loss tensor and notify all cores that epoch ended for this chunk
                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)

                loss_log = xm.all_reduce(xm.REDUCE_SUM, loss_sum) if stage == 1 else loss_sum

                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")

                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    xm.master_print("-" * 60)
                    xm.master_print(f"STAGE {stage} | CHUNK {chunk_idx + 1} | EPOCH {epoch + 1}")
                    xm.master_print(f"  LR:         {current_lr:.2e}")
                    xm.master_print(f"  Total Loss: {loss_sum / num_cores:.4f}")
                    if stage == 1:
                        xm.master_print(f"  Cls Loss:   {loss_log / num_cores:.4f}")
                    else:
                        xm.master_print(f"  Halt Loss:  {loss_log / num_cores:.4f}")

                    def format_sample(data, name):
                        if data is None:
                            return f"  {name}: No sample found."
                        out = [f"  > {name} (Label {data['lbl'].item()}):"]
                        h_probs = torch.sigmoid(data['halt'])
                        out.append(f"    HALT Probs: {[f'{p:.2f}' for p in h_probs.tolist()]}")
                        return "\n".join(out)

                    xm.master_print("  DIAGNOSTICS:")
                    xm.master_print(format_sample(diag_sample_pos, "Sample POS"))
                    xm.master_print(format_sample(diag_sample_neg, "Sample NEG"))

            xm.rendezvous(f"chunk_end_st{stage}_ch{chunk_idx}")

    xm.rendezvous("ready_to_save_final")
    save_path = os.path.expanduser("~/SIGNLL/final_model_stage2_gated.pt")

    if rank == 0:
        xm.master_print(f"Saving final model: {save_path}")
        torch.save(model.state_dict(), save_path)
        xm.master_print("✅ Training Complete with Gated Entropy.")

    xm.rendezvous("save_complete_safe_exit")


def _mp_fn(rank, flags):
    try:
        torch.set_default_tensor_type('torch.FloatTensor')
        train_loop(rank, flags)
    except Exception as e:
        print(f"[Core {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 5,
        "samples_per_shard": 39000
    }

    print("Starting Stage-2 Training with Gated Entropy (DistributedSampler, TPU-safe).")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')
