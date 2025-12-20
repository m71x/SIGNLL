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

from controller_model import Controller # compute_q_from_h removed
from training_data_download import training_data_download

# =========================================================================
# SAM OPTIMIZER CLASS
# =========================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # Do the actual "w - lr * grad" update
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
                    p=2
               )
        return norm

# =========================================================================
# EVALUATION FUNCTION (Unchanged)
# =========================================================================
def evaluate_model(rank, model, chunk_idx, threshold, batch_size, samples_per_shard):
    """
    Tests the model on a specific chunk using an early-exit strategy.
    Runs ONLY on Device 0.
    """
    if rank != 0:
        return

    device = xm.xla_device()
    
    xm.master_print(f"\n{'*'*80}")
    xm.master_print(f"*** STARTING EVALUATION ON CHUNK {chunk_idx} (Device 0 Only) ***")
    xm.master_print(f"{'*'*80}")

    model.eval()
    
    current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
    data = training_data_download(
        core_id=0, 
        filename=current_chunk_filename,
        max_entries=samples_per_shard
    )
    
    if data is None:
        xm.master_print(f"[Core {rank}] Failed to load test data for chunk {chunk_idx}")
        return

    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]
    
    dataset = TensorDataset(teacher_cls_full, teacher_label_full)
    sampler = SequentialSampler(dataset)
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2
    )
    
    total_samples = 0
    total_correct = 0
    layer_exit_counts_cpu = torch.zeros(24, dtype=torch.float32)

    with torch.no_grad():
        for i, (teacher_cls, teacher_label) in enumerate(data_loader):
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            
            halting_logits, class_logits, _ = model(teacher_cls)
            
            h_probs = torch.sigmoid(halting_logits)
            threshold_mask = (h_probs > threshold)
            
            exit_indices = torch.argmax(threshold_mask.long(), dim=1)
            row_has_exit = threshold_mask.any(dim=1)
            exit_indices[~row_has_exit] = 23
            
            batch_indices = torch.arange(class_logits.size(0), device=device)
            selected_logits = class_logits[batch_indices, exit_indices]
            predictions = torch.argmax(selected_logits, dim=-1)
            
            correct_tensor = (predictions == teacher_label).sum()
            total_correct += correct_tensor.item() 
            total_samples += teacher_label.size(0)
            
            exit_indices_cpu = exit_indices.cpu()
            unique_exits, counts = torch.unique(exit_indices_cpu, return_counts=True)
            layer_exit_counts_cpu.index_add_(0, unique_exits, counts.float())
            
            xm.mark_step()
            
            if i % 100 == 0:
                print(f"[Eval] Processed batch {i}...")
            
    accuracy = (total_correct / total_samples) * 100.0
    
    # --- Statistics Calculation ---
    layers = torch.arange(24, dtype=torch.float32)
    avg_exit_layer = (layer_exit_counts_cpu * layers).sum() / total_samples
    variance = (layer_exit_counts_cpu * (layers - avg_exit_layer).pow(2)).sum() / total_samples
    std_exit_layer = torch.sqrt(variance)
    
    # --- MAD Calculation ---
    counts_int = layer_exit_counts_cpu.long()
    all_exit_layers = torch.repeat_interleave(layers, counts_int)
    med_exit_layer = all_exit_layers.median()
    abs_dev = torch.abs(all_exit_layers - med_exit_layer)
    mad_exit_layer = abs_dev.median()
    
    xm.master_print(f"RESULTS FOR CHUNK {chunk_idx} (Threshold: {threshold}):")
    xm.master_print(f"  Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")
    xm.master_print(f"  Average Exit Layer: {avg_exit_layer:.2f} +/- {std_exit_layer:.2f} (0-23)")
    xm.master_print(f"  Median Exit Layer:  {med_exit_layer:.2f} (MAD: {mad_exit_layer:.2f})")
    
    # --- Histogram Log (NEW) ---
    xm.master_print(f"  Exit Layer Distribution (0-23): {layer_exit_counts_cpu.long().tolist()}")
    
    xm.master_print(f"{'*'*80}\n")

    model.train() 

# =========================================================================
# TRAINING LOOP 
# =========================================================================
def train_loop(rank, flags):
    device = xm.torch_xla.device()
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
    
    # Initial Rendezvous
    xm.rendezvous("weights_synced") 

    # Calculate step counts
    total_samples = flags["samples_per_shard"]
    num_batches_per_chunk = total_samples // flags["batch_size"]
    total_steps_stage_2 = flags["epochs"] * 29 * num_batches_per_chunk
    global_step = 0
    start_time = time.time()

    # Placeholders for sample diagnostics (Rank 0 only)
    diag_sample_pos = None
    diag_sample_neg = None

    # =========================================================================
    # STAGE LOOP: 1 -> Backbone/Classifiers, 2 -> Halting Heads
    # =========================================================================
    for stage in [1, 2]:
        
        # --- PHASE SETUP ---
        if stage == 1:
            stage_name = "STAGE 1: Backbone & Classifiers (Halting IGNORED)"
            for param in model.parameters(): param.requires_grad = True
            for param in model.halting_heads.parameters(): param.requires_grad = False
            
        elif stage == 2:
            stage_name = "STAGE 2: Halting Heads Only (Backbone & Classifiers FROZEN)"
            for param in model.parameters(): param.requires_grad = False
            for param in model.halting_heads.parameters(): param.requires_grad = True

        if rank == 0:
            xm.master_print(f"\n{'#'*80}")
            xm.master_print(f"STARTING {stage_name}")
            xm.master_print(f"{'#'*80}")
        
        # --- OPTIMIZER FIX: Parameter Groups ---
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        # Initialize SAM with AdamW as the base optimizer
        optimizer = SAM(model_params, optim.AdamW, rho=0.05, lr=flags["lr"], weight_decay=1e-2)

        # --- SCHEDULER SETUP ---
        total_steps_in_stage = 28 * flags["epochs"] * num_batches_per_chunk
        T_0 = total_steps_in_stage // 6
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, # SAM optimizer wraps the base, scheduler steps on SAM
            T_0=T_0, 
            T_mult=2,   
            eta_min=1e-6
        )
        if rank == 0:
            xm.master_print(f"Scheduler Initialized: CosineAnnealingWarmRestarts with T_0={T_0} steps")

        for chunk_idx in range(28): 
            current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
            
            if rank == 0:
                xm.master_print(f"Stage {stage} | Chunk {chunk_idx + 1}/29 | Loading {current_chunk_filename}")

            # --- Load Data ---
            data = training_data_download(
                core_id=rank,
                filename=current_chunk_filename,
                max_entries=flags["samples_per_shard"]
            )
            
            if data is None: raise RuntimeError(f"[Core {rank}] Failed load chunk {chunk_idx}")

            teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
            teacher_label_full = torch.from_numpy(data['classifications']).long()
            
            # --- START ENHANCEMENT 9: Soft Targets Loading/Synthesis ---
            if 'teacher_logits' in data:
                t_logits = torch.from_numpy(data['teacher_logits']).float()
                T_distill = 2.0
                teacher_log_probs_full = F.log_softmax(t_logits / T_distill, dim=-1)
            else:
                num_classes = 2
                smoothing = 0.1
                t_one_hot = torch.zeros(teacher_label_full.size(0), num_classes).scatter_(1, teacher_label_full.unsqueeze(1), 1)
                teacher_probs_full = t_one_hot * (1.0 - smoothing) + (smoothing / num_classes)
                teacher_log_probs_full = torch.log(teacher_probs_full)
                
                if rank == 0 and chunk_idx == 0:
                    xm.master_print("  [Note] 'teacher_logits' not found. Using synthesized Soft Targets (Label Smoothing=0.1).")

            if teacher_cls_full.shape[1] == 25:
                teacher_cls_full = teacher_cls_full[:, 1:25, :]
            
            # --- Data Slicing (10/32) ---
            N_total_local = teacher_cls_full.shape[0]
            N_target = (N_total_local // num_cores) * 32 

            teacher_cls_full = teacher_cls_full[:N_target]
            teacher_label_full = teacher_label_full[:N_target]
            teacher_log_probs_full = teacher_log_probs_full[:N_target] 

            if rank == 0:
                xm.master_print(f"Data Sliced: Using {N_target}/{N_total_local} samples ({N_target/N_total_local:.2%}) for 18.75% utilization.")

            # Class Weighting
            neg_samples = (teacher_label_full == 0).sum().item()
            pos_samples = (teacher_label_full == 1).sum().item()
            pos_weight_val = neg_samples / (pos_samples + 1e-6)
            pos_weight_tensor = torch.tensor([pos_weight_val]).float().to(device)

            bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor).to(device)
            
            dataset = TensorDataset(teacher_cls_full, teacher_label_full, teacher_log_probs_full)
            sampler = RandomSampler(dataset)
            
            data_loader = DataLoader(dataset, sampler=sampler, batch_size=flags["batch_size"], drop_last=True, num_workers=2)
            
            # --- Epoch Loop ---
            for epoch in range(flags["epochs"]):
                model.train()
                
                diag_sample_pos = None
                diag_sample_neg = None

                for batch_idx, (teacher_cls, teacher_label, teacher_log_probs) in enumerate(data_loader):
                    if stage == 2: global_step += 29 
                    
                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    teacher_log_probs = teacher_log_probs.to(device)
                    
                    # --- COMPUTE LOSS CLOSURE ---
                    # We define this inner function so we can run it twice (once for initial grad, once for SAM)
                    def compute_loss_step():
                        # Forward Pass
                        halting_logits, class_logits, z = model(teacher_cls) 
                        
                        # --- LOSS CALCULATION ---
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
                        kl_elementwise = teacher_log_probs_expanded.exp() * (teacher_log_probs_expanded - student_log_probs)
                        loss_soft = kl_elementwise.sum(dim=-1)

                        alpha = 0.5
                        ce_per_layer = (alpha * loss_hard) + ((1 - alpha) * loss_soft)

                        loss_contrast = F.mse_loss(z[:, 1:, :], z[:, :-1, :])

                        if stage == 1:
                            loss_cls = ce_per_layer.mean()
                            loss = (loss_cls * 2) + (0.1 * loss_contrast)
                            
                            # For logging (return tuple)
                            return loss, loss_cls, torch.tensor(0.0), halting_logits, class_logits

                        elif stage == 2:
                            # --- INDEPENDENT HALTING LOSS ---
                            predictions = torch.argmax(class_logits, dim=-1)
                            is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                            
                            h = torch.sigmoid(halting_logits)
                            
                            # Class-Aware Rebalancing
                            n_pos = (teacher_label == 1).sum().float()
                            n_neg = (teacher_label == 0).sum().float()
                            neg_weight_val = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)
                            sample_weights = torch.ones_like(h)
                            sample_weights[teacher_label == 0] = neg_weight_val.item()
                            
                            loss_halt = F.binary_cross_entropy(h, is_correct, weight=sample_weights)
                            
                            # Entropy Regularization
                            entropy_weight = 0.0025 
                            h_entropy = - (h * (h + 1e-9).log() + (1 - h) * (1 - h + 1e-9).log())
                            loss_entropy = - entropy_weight * h_entropy.mean()
                            
                            # Orthogonality Regularization
                            ortho_weight = 0.01 
                            head_weights = torch.cat([head.weight for head in model.halting_heads], dim=0)
                            head_weights_norm = F.normalize(head_weights, p=2, dim=1)
                            sim_matrix = torch.mm(head_weights_norm, head_weights_norm.t())
                            identity = torch.eye(model.L, device=device)
                            loss_ortho = ((sim_matrix - identity) ** 2).sum() * ortho_weight
                            
                            loss = loss_halt + loss_entropy + loss_ortho
                            
                            return loss, torch.tensor(0.0), loss_halt, halting_logits, class_logits

                    # --- SAM OPTIMIZATION STEP (Manual for XLA) ---
                    # 1. First Forward & Backward
                    loss, loss_cls_val, loss_halt_val, halting_logits, class_logits = compute_loss_step()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Sync gradients across TPU cores before perturbation
                    xm.reduce_gradients(optimizer) 
                    
                    # 2. Perturb Weights (Ascent)
                    optimizer.first_step(zero_grad=True)
                    
                    # 3. Second Forward & Backward (at perturbed state)
                    loss_2, _, _, _, _ = compute_loss_step()
                    loss_2.backward()
                    
                    # Sync gradients again
                    xm.reduce_gradients(optimizer)
                    
                    # 4. Update Weights (Descent)
                    optimizer.second_step(zero_grad=True)
                    
                    # Scheduler Step
                    scheduler.step()
                    xm.mark_step()
                    
                    # --- CAPTURE DIAGNOSTICS (Rank 0) ---
                    if rank == 0 and batch_idx == 0:
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

                # --- End of Epoch Logging ---
                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                
                if stage == 1:
                    loss_log = xm.all_reduce(xm.REDUCE_SUM, loss_cls_val)
                else:
                    loss_log = xm.all_reduce(xm.REDUCE_SUM, loss_halt_val) 
                
                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")
                
                if rank == 0:
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    xm.master_print("-" * 60)
                    xm.master_print(f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}")
                    xm.master_print(f"  LR:         {current_lr:.2e}")
                    xm.master_print(f"  Total Loss: {loss_sum / num_cores:.4f}")
                    if stage == 1:
                        xm.master_print(f"  Cls Loss:   {loss_log / num_cores:.4f}")
                    else:
                        xm.master_print(f"  Halt Loss:  {loss_log / num_cores:.4f} (W-BCE + Ent + Ortho + SAM)")
                    
                    def format_sample(data, name):
                        if data is None: return f"  {name}: No sample found in first batch."
                        out = [f"  > {name} (Label {data['lbl'].item()}):"]
                        if stage == 1:
                            probs = torch.softmax(data['cls'], dim=-1) 
                            cls1_probs = probs[:, 1]
                            out.append(f"    CLS Probs (Class 1): {[f'{p:.2f}' for p in cls1_probs.tolist()]}")
                        else:
                            h_probs = torch.sigmoid(data['halt'])
                            out.append(f"    HALT Probs: {[f'{p:.2f}' for p in h_probs.tolist()]}")
                        return "\n".join(out)

                    xm.master_print("  DIAGNOSTICS (Layer 0->23):")
                    xm.master_print(format_sample(diag_sample_pos, "Sample POS"))
                    xm.master_print(format_sample(diag_sample_neg, "Sample NEG"))

            if (chunk_idx + 1) % 5 == 0 and rank == 0:
                torch.save(model.state_dict(), f"/tmp/controller_stage{stage}_chunk{chunk_idx+1}.pt")
                xm.master_print("Saved Checkpoint.")

            xm.rendezvous(f"chunk_end_st{stage}_ch{chunk_idx}")

    if rank == 0:
        torch.save(model.state_dict(), "/tmp/controller_final_stage2.pt")
        xm.master_print("✅ Training Complete.")

    test_chunk = flags.get("test_chunk", 29) 
    evaluate_model(rank, model, test_chunk, 0.5, flags["batch_size"], flags["samples_per_shard"])
    evaluate_model(rank, model, test_chunk, 0.6, flags["batch_size"], flags["samples_per_shard"])
    evaluate_model(rank, model, test_chunk, 0.7, flags["batch_size"], flags["samples_per_shard"])
    evaluate_model(rank, model, test_chunk, 0.8, flags["batch_size"], flags["samples_per_shard"])
    evaluate_model(rank, model, test_chunk, 0.95, flags["batch_size"], flags["samples_per_shard"])
    
    xm.rendezvous("final_check")

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
        "batch_size": 32,   
        "lambda_start": 0.0001,
        "lambda_target": 0.003,
        "epochs": 5,
        "samples_per_shard": 39000, 
        "test_chunk": 29, 
        "test_threshold": 0.8
    }  
    
    print("Starting Two-Stage XLA Job.")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')