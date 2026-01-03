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

from controller_gru import Controller
from training_data_download import training_data_download

# =========================================================================
# HELPER: Supervised Contrastive Loss
# =========================================================================
def supervised_contrastive_loss(features, labels, temperature=0.1):
    """
    Args:
        features: Hidden vectors [Batch_Size, Dim] (Should be normalized)
        labels: Ground truth labels [Batch_Size]
        temperature: Scalar temperature parameter
    """
    device = features.device
    batch_size = features.shape[0]
    
    labels = labels.contiguous().view(-1, 1)
    # Mask of shape [B, B]: 1 if i and j have same label, 0 otherwise
    mask = torch.eq(labels, labels.T).float().to(device)

    # Compute similarity matrix (Cosine Similarity since features are normalized)
    # [B, B]
    anchor_dot_contrast = torch.matmul(features, features.T) / temperature

    # For numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # Mask out self-contrast (diagonal)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    
    # Mask to ignore self-comparisons in the positive mask
    mask = mask * logits_mask

    # Compute Log-Sum-Exp
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

    # Compute mean log-likelihood for positive pairs
    # Sum over j (positives), then divide by count of positives
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

    # Loss is negative mean log likelihood
    return -mean_log_prob_pos.mean()

# =========================================================================
# HELPER: Monotonicity Loss
# =========================================================================
def monotonicity_loss(class_logits, labels, margin=0.0):
    """
    Penalizes the model if the loss at step t is greater than step t-1.
    Enforces that performance should not degrade as we process more layers.
    Args:
        class_logits: [B, L, Num_Classes]
        labels: [B]
    """
    B, L, C = class_logits.shape
    device = class_logits.device
    
    # Expand labels: [B, L]
    labels_exp = labels.unsqueeze(1).expand(-1, L)
    
    # Compute CE loss per step: [B, L]
    # We use reduction='none' to preserve step-wise loss structure
    ce_per_step = F.cross_entropy(
        class_logits.reshape(-1, C), 
        labels_exp.reshape(-1), 
        reduction='none'
    ).view(B, L)
    
    # Calculate difference: Loss(t) - Loss(t-1)
    # Ideally, Loss(t) < Loss(t-1), so diff should be negative.
    # If diff is positive (loss went UP), we penalize it.
    loss_t = ce_per_step[:, 1:]
    loss_prev = ce_per_step[:, :-1]
    
    diff = loss_t - loss_prev
    
    # Penalize violations (where loss increased)
    penalty = F.relu(diff + margin)
    
    return penalty.mean()

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
        num_classes=2,
        d_halt_hidden=64
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
            
            # Freeze Recurrent Modules
            for m in [model.halting_gru, model.halt_ln, model.halting_proj]:
                for p in m.parameters():
                    p.requires_grad = False
            
            for p in model.entropy_gate_module.parameters():
                p.requires_grad = False
        else:
            stage_name = "STAGE 2: Recurrent Halting + Diversity (Backbone FROZEN)"
            for p in model.parameters():
                p.requires_grad = False
            
            # Unfreeze Recurrent Modules
            for m in [model.halting_gru, model.halt_ln, model.halting_proj]:
                for p in m.parameters():
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
        T_0 = max(1, total_steps // 6)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
        )

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
                num_replicas=1, 
                rank=0,         
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

                diag_sample_pos = None
                diag_sample_neg = None

                for batch_idx, (x, y) in enumerate(loader):
                    x = x.to(device)
                    y = y.to(device)

                    # Capture return values (now 4 items)
                    halting_logits, class_logits, z, saved_states = model(x)

                    # -----------------------------
                    # Stage-specific losses
                    # -----------------------------
                    if stage == 1:
                        # 1. Calculate Weights
                        n_pos = (y == 1).sum().float()
                        n_neg = (y == 0).sum().float()
                        pos_weight_val = n_neg / (n_pos + 1e-6)
                        pos_weight_tensor = torch.tensor([pos_weight_val], device=device)

                        # 2. Hard Loss
                        labels_expanded = y.float().unsqueeze(1).expand(-1, class_logits.shape[1])
                        class_logits_positive = class_logits[:, :, 1]
                        student_log_probs = F.log_softmax(class_logits, dim=-1)
                        
                        loss_hard = F.binary_cross_entropy_with_logits(
                            class_logits_positive, 
                            labels_expanded, 
                            pos_weight=pos_weight_tensor, 
                            reduction='none'
                        )

                        # 3. Soft Loss
                        num_classes = 2
                        smoothing = 0.1
                        t_one_hot = torch.zeros(y.size(0), num_classes, device=device).scatter_(1, y.unsqueeze(1), 1)
                        teacher_probs = t_one_hot * (1.0 - smoothing) + (smoothing / num_classes)
                        teacher_log_probs = torch.log(teacher_probs.clamp(min=1e-10))
                        
                        teacher_log_probs_expanded = teacher_log_probs.unsqueeze(1).expand(-1, class_logits.shape[1], -1)
                        kl_elementwise = teacher_log_probs_expanded.exp() * (
                            teacher_log_probs_expanded - student_log_probs
                        )
                        loss_soft = kl_elementwise.sum(dim=-1)

                        # 4. SUPERVISED CONTRASTIVE LOSS
                        # z shape: [B, L, D]
                        # We pool across L to get a stable "trajectory representation" [B, D]
                        z_pooled = torch.mean(z, dim=1)
                        features = F.normalize(z_pooled, dim=1)
                        loss_contrast = supervised_contrastive_loss(features, y, temperature=0.1)

                        # 5. MONOTONICITY LOSS
                        loss_mono = monotonicity_loss(class_logits, y)

                        alpha = 0.5
                        ce_per_layer = (alpha * loss_hard) + ((1 - alpha) * loss_soft)
                        loss_cls = ce_per_layer.mean()
                        
                        # Combined Loss: CE + SupCon + Monotonicity
                        loss = (loss_cls * 2) + (0.1 * loss_contrast) + (0.1 * loss_mono)

                        if rank == 0 and batch_idx == 0:
                            with torch.no_grad():
                                def extract_sample(label_val):
                                    indices = (y == label_val).nonzero(as_tuple=True)[0]
                                    if indices.numel() > 0:
                                        idx = indices[0]
                                        return {
                                            "cls": class_logits[idx].detach().cpu(),
                                            "lbl": y[idx].detach().cpu()
                                        }
                                    return None
                                diag_sample_pos = extract_sample(1)
                                diag_sample_neg = extract_sample(0)

                    else:
                        # STAGE 2: RECURRENT HALTING + DIVERSITY
                        preds = torch.argmax(class_logits, dim=-1)
                        is_correct = (preds == y.unsqueeze(1)).float()

                        # 1. Class-Aware Rebalancing
                        n_pos = (y == 1).sum().float()
                        n_neg = (y == 0).sum().float()
                        neg_weight_val = ((n_pos / (n_neg + 1e-6)).clamp(min=1.0)) * 8.0

                        sample_weights = torch.ones_like(halting_logits)
                        sample_weights[y == 0] = neg_weight_val.item()

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

                        # 4. INNOVATION / DIVERSITY LOSS
                        # saved_states: List[24] of [B, 64] -> Stack: [B, 24, 64]
                        h_stack = torch.stack(saved_states, dim=1)
                        
                        # Shift to compare t vs t-1. Prepend zeros for first step
                        h_prev = torch.cat([torch.zeros_like(h_stack[:, :1, :]), h_stack[:, :-1, :]], dim=1)
                        
                        # A. Stagnation Penalty: Penalize high Cosine Similarity
                        cos_sim = F.cosine_similarity(h_stack, h_prev, dim=-1)
                        loss_diversity = F.relu(cos_sim - 0.9).mean() * 0.1
                        
                        # B. Magnitude Penalty: Force change
                        delta_mag = torch.norm(h_stack - h_prev, p=2, dim=-1)
                        loss_innovation = F.relu(0.1 - delta_mag).mean() * 0.1
                        
                        loss = loss_halt + loss_entropy + loss_diversity + loss_innovation

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

                    if stage == 1:
                        xm.master_print(f"  Cls Loss:   {loss_sum / num_cores:.4f}")
                    else:
                        xm.master_print(f"  Halt Loss:  {loss_sum / num_cores:.4f}")

                    def format_sample(data, name):
                        if data is None:
                            return f"  {name}: No sample found."
                        out = [f"  > {name} (Label {data['lbl'].item()}):"]
                        if stage == 1:
                            probs = torch.softmax(data["cls"], dim=-1)
                            cls1_probs = probs[:, 1]
                            out.append(
                                f"    CLS Probs (Class 1): {[f'{p:.2f}' for p in cls1_probs.tolist()]}"
                            )
                        else:
                            h_probs = torch.sigmoid(data["halt"])
                            out.append(
                                f"    HALT Probs: {[f'{p:.2f}' for p in h_probs.tolist()]}"
                            )
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

    print("Starting Stage-1 + Stage-2 Training (TPU-safe).")
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method="fork")