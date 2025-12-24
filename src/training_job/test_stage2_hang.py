import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
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
    if rank == 0: xm.master_print("Synchronizing initial weights...")
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    
    xm.rendezvous("weights_synced") 

    # Calculate step counts
    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    global_step = 0

    diag_sample_pos = None
    diag_sample_neg = None

    # =========================================================================
    # STAGE LOOP: SKIPPING DIRECTLY TO STAGE 2 FOR TESTING
    # =========================================================================
    for stage in [2]: 
        
        # --- PHASE SETUP ---
        if stage == 1:
            stage_name = "STAGE 1: Backbone & Classifiers (Halting & Gates FROZEN)"
            for param in model.parameters(): 
                param.requires_grad = True
            for param in model.halting_heads.parameters(): 
                param.requires_grad = False
            for param in model.entropy_gate_module.parameters(): 
                param.requires_grad = False
            
        elif stage == 2:
            stage_name = "STAGE 2: Halting Heads + Entropy Gates (Backbone FROZEN)"
            for param in model.parameters(): 
                param.requires_grad = False
            for param in model.halting_heads.parameters(): 
                param.requires_grad = True
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
        total_steps_in_stage = 28 * flags["epochs"] * num_batches_per_chunk
        T_0 = total_steps_in_stage // 4
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0, 
            T_mult=2,   
            eta_min=1e-6
        )
        if rank == 0:
            xm.master_print(f"Scheduler: CosineAnnealingWarmRestarts, T_0={T_0} steps")

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
            
            # --- Data Slicing ---
            N_total_local = teacher_cls_full.shape[0]
            N_target = (N_total_local // num_cores) * 32

            teacher_cls_full = teacher_cls_full[:N_target]
            teacher_label_full = teacher_label_full[:N_target]
            teacher_log_probs_full = teacher_log_probs_full[:N_target] 

            # Stage 1 Pos Weight (Static)
            neg_samples = (teacher_label_full == 0).sum().item()
            pos_samples = (teacher_label_full == 1).sum().item()
            pos_weight_val = neg_samples / (pos_samples + 1e-6)
            pos_weight_tensor = torch.tensor([pos_weight_val]).float().to(device)

            bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor).to(device)
            
            dataset = TensorDataset(teacher_cls_full, teacher_label_full, teacher_log_probs_full)
            sampler = RandomSampler(dataset)
            
            data_loader = DataLoader(
                dataset, sampler=sampler, batch_size=flags["batch_size"], 
                drop_last=True, num_workers=2
            )
            
            # --- Epoch Loop ---
            for epoch in range(flags["epochs"]):
                model.train()
                diag_sample_pos = None
                diag_sample_neg = None

                for batch_idx, (teacher_cls, teacher_label, teacher_log_probs) in enumerate(data_loader):
                    if stage == 2: 
                        global_step += 1
                    
                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    teacher_log_probs = teacher_log_probs.to(device)
                    
                    # --- STAGE 1: Standard Training ---
                    if stage == 1:
                        halting_logits, class_logits, z = model(teacher_cls) 
                        
                        labels = teacher_label.float().unsqueeze(1).expand(-1, L)
                        
                        if class_logits.size(-1) == 2:
                            class_logits_positive = class_logits[:, :, 1]
                            student_log_probs = F.log_softmax(class_logits, dim=-1)
                        else:
                            class_logits_positive = class_logits.squeeze(-1)
                            student_log_probs = F.log_softmax(
                                torch.stack([-class_logits_positive, class_logits_positive], dim=-1),
                                dim=-1
                            )
                        
                        loss_hard = bce_loss_fn(class_logits_positive, labels)

                        teacher_log_probs_expanded = teacher_log_probs.unsqueeze(1).expand(-1, L, -1)
                        kl_elementwise = teacher_log_probs_expanded.exp() * (
                            teacher_log_probs_expanded - student_log_probs
                        )
                        loss_soft = kl_elementwise.sum(dim=-1)

                        alpha = 0.5
                        ce_per_layer = (alpha * loss_hard) + ((1 - alpha) * loss_soft)
                        loss_contrast = F.mse_loss(z[:, 1:, :], z[:, :-1, :])

                        loss_cls = ce_per_layer.mean()
                        loss = (loss_cls * 2) + (0.1 * loss_contrast)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        xm.optimizer_step(optimizer)
                        scheduler.step()
                        xm.mark_step()
                        
                        # Diagnostics (Rank 0)
                        if rank == 0 and batch_idx == 0:
                            xm.mark_step() # Ensure step is done before pulling data
                            with torch.no_grad():
                                d_halt, d_cls, _ = model(teacher_cls)
                                def extract_sample(label_val):
                                    indices = (teacher_label == label_val).nonzero(as_tuple=True)[0]
                                    if indices.numel() > 0:
                                        idx = indices[0]
                                        return {
                                            'cls': d_cls[idx].detach().cpu(),
                                            'halt': d_halt[idx].detach().cpu(),
                                            'lbl': teacher_label[idx].detach().cpu()
                                        }
                                    return None
                                diag_sample_pos = extract_sample(1)
                                diag_sample_neg = extract_sample(0)

                    # --- STAGE 2: Standard Training ---
                    elif stage == 2:
                        # Forward pass
                        halting_logits, class_logits, z = model(teacher_cls)
                        
                        predictions = torch.argmax(class_logits, dim=-1)
                        is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                        
                        # CLASS-AWARE REBALANCING (STABILIZED)
                        # 1. Calculate counts
                        n_pos = (teacher_label == 1).sum().float()
                        n_neg = (teacher_label == 0).sum().float()
                        
                        # 2. Ratio with MAX CLAMP
                        # FIX: Added max=50.0 to prevent gradient explosion on sparse batches
                        neg_weight_tensor = (n_pos / (n_neg + 1e-6)).clamp(min=1.0, max=50.0)
                        
                        # 3. Create weight tensor [B, 1] using Arithmetic (No torch.where)
                        mask_neg_float = (teacher_label == 0).float().unsqueeze(1) 
                        weights_b1 = 1.0 + (mask_neg_float * (neg_weight_tensor - 1.0))
                        
                        # 4. Explicitly expand to [B, L] to match logits shape exactly
                        sample_weights = weights_b1.expand(-1, L)
                        
                        # Halting Loss
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
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        xm.optimizer_step(optimizer)
                        scheduler.step()
                        xm.mark_step()

                        # Diagnostics (Rank 0)
                        if rank == 0 and batch_idx == 0:
                            xm.mark_step() # Ensure step completion before diagnostics
                            with torch.no_grad():
                                d_halt, d_cls, _ = model(teacher_cls)
                                def extract_sample(label_val):
                                    indices = (teacher_label == label_val).nonzero(as_tuple=True)[0]
                                    if indices.numel() > 0:
                                        idx = indices[0]
                                        return {
                                            'cls': d_cls[idx].detach().cpu(),
                                            'halt': d_halt[idx].detach().cpu(),
                                            'lbl': teacher_label[idx].detach().cpu()
                                        }
                                    return None
                                diag_sample_pos = extract_sample(1)
                                diag_sample_neg = extract_sample(0)

                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                
                if stage == 1:
                    loss_log = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
                else:
                    loss_log = loss_sum 
                
                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")
                
                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    xm.master_print("-" * 60)
                    xm.master_print(f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}")
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
                        if stage == 1:
                            probs = torch.softmax(data['cls'], dim=-1) 
                            cls1_probs = probs[:, 1]
                            out.append(f"    CLS Probs (Class 1): {[f'{p:.2f}' for p in cls1_probs.tolist()]}")
                        else:
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
    
    print("Starting Two-Stage Training with Gated Entropy (TEST MODE: STAGE 2 ONLY).")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')