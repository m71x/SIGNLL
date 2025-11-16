import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# (1) Import new classes
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

def train_loop(rank, flags):
    device = xm.torch_xla.device()
    num_cores = xm.xrt_world_size()

    if rank == 0:
        xm.master_print(f"Detected {num_cores} active cores")
        xm.master_print(f"Starting training with {flags['samples_per_shard']} samples per core")

    # --- Load Data Once (On CPU!) ---
    data = training_data_download(
        core_id=rank,
        filename=flags["chunk_filename"],
        max_entries=flags["samples_per_shard"]
    )
    
    if data is None:
        raise RuntimeError(f"[Core {rank}] Failed to load training data")

    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    # Handle layer slicing
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]
    elif teacher_cls_full.shape[1] != 24:
        raise ValueError(f"Unexpected number of layers: {teacher_cls_full.shape[1]}")

    # === Calculate and Apply Class Weighting ===
    neg_samples_count = (teacher_label_full == 0).sum().item() 
    pos_samples_count = (teacher_label_full == 1).sum().item() 
    
    pos_weight_value = neg_samples_count / pos_samples_count
    pos_weight_tensor = torch.tensor([pos_weight_value]).float()
    
    print(f"[Core {rank}] Data Shard Check: {neg_samples_count} samples have Label 0 (Negative).")
    if rank == 0:
        xm.master_print(f"[Core {rank}] Calculated Positive Weight (pos_weight): {pos_weight_value:.4f}")
    
    num_samples = teacher_cls_full.shape[0]

    # --- Create Dataset and DataLoader ---
    dataset = TensorDataset(teacher_cls_full, teacher_label_full)
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_cores,
        rank=rank,
        shuffle=True
    )

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=flags["batch_size"],
        drop_last=True,
        num_workers=2
    )

    xm.rendezvous("data_prepared")

    if rank == 0:
        xm.master_print(f"All {num_cores} cores have loaded data successfully")
        xm.master_print(f"CLS tokens shape (CPU): {teacher_cls_full.shape}")

    # --- Initialize Model ---
    print(f"[Core {rank}] Initializing model with {flags['transformer_layers']} transformer layers...")
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],  # Now 8 layers
        num_classes=2
    ).to(device)
    
    # ========== SYNCHRONIZE INITIAL WEIGHTS ==========
    if rank == 0:
        xm.master_print("Synchronizing initial weights across all cores...")
    
    for param in model.parameters():
        if rank == 0:
            param.data = param.data * num_cores
        else:
            param.data = param.data * 0
        
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data)
        param.data = param.data / num_cores
    
    xm.mark_step()
    xm.rendezvous("weights_synced")
    
    if rank == 0:
        xm.master_print("✅ Initial weights synchronized across all cores")

    optimizer = optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-2)
    
    pos_weight_device = pos_weight_tensor.to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss(
        reduction="none", 
        pos_weight=pos_weight_device
    ).to(device)

    # --- Training Configuration ---
    num_epochs = flags["epochs"]
    batch_size = flags["batch_size"]
    lambda_start, lambda_target = flags["lambda_start"], flags["lambda_target"]
    num_batches_per_epoch = len(data_loader)
    total_steps = num_epochs * num_batches_per_epoch

    if rank == 0:
        xm.master_print(f"Training config: {num_epochs} epochs, {num_batches_per_epoch} batches/epoch")
        xm.master_print(f"Total global samples: (approx) {num_samples * num_cores}")
        xm.master_print(f"Total steps: {total_steps}")
    
    global_step = 0
    start_time = time.time()
    model.train()

    # Variables to hold diagnostic data
    sample_logits_cpu = None
    sample_label_cpu = None
    sample_logits_pos_cpu = None
    sample_label_pos_cpu = None
    sample_logits_neg_cpu = None
    sample_label_neg_cpu = None

    # --- Full Training Loop ---
    for epoch in range(num_epochs):
        if rank == 0:
            xm.master_print(f"\n{'='*80}")
            xm.master_print(f"EPOCH {epoch + 1}/{num_epochs}")
            xm.master_print(f"{'='*80}")
        
        for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
            
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            
            # --- Forward Pass ---
            halting_logits, class_logits, _ = model(teacher_cls)
            
            # === NaN Detection (BEFORE sigmoid) ===
            if torch.isnan(halting_logits).any() or torch.isinf(halting_logits).any():
                if rank == 0:
                    xm.master_print(f"[Core {rank}] ⚠️  NaN/Inf in halting_logits at step {global_step}")
                    xm.master_print(f"  Range: [{halting_logits.min()}, {halting_logits.max()}]")
                raise ValueError("NaN detected in halting_logits")
            
            if torch.isnan(class_logits).any() or torch.isinf(class_logits).any():
                if rank == 0:
                    xm.master_print(f"[Core {rank}] ⚠️  NaN/Inf in class_logits at step {global_step}")
                raise ValueError("NaN detected in class_logits")
            
            # === Capture diagnostic data ===
            if rank == 0 and batch_idx == 0:
                sample_logits_cpu = class_logits[0].detach().cpu()
                sample_label_cpu = teacher_label[0].detach().cpu()
                
                positive_indices = (teacher_label == 1).nonzero(as_tuple=True)[0]
                negative_indices = (teacher_label == 0).nonzero(as_tuple=True)[0]

                if positive_indices.numel() > 0:
                    pos_idx = positive_indices[0]
                    sample_logits_pos_cpu = class_logits[pos_idx].detach().cpu() 
                    sample_label_pos_cpu = teacher_label[pos_idx].detach().cpu()
                else:
                    sample_logits_pos_cpu = None
                    sample_label_pos_cpu = None

                if negative_indices.numel() > 0:
                    neg_idx = negative_indices[0]
                    sample_logits_neg_cpu = class_logits[neg_idx].detach().cpu()
                    sample_label_neg_cpu = teacher_label[neg_idx].detach().cpu()
                else:
                    sample_logits_neg_cpu = None
                    sample_label_neg_cpu = None
            
            # --- Loss Calculation ---
            h = torch.sigmoid(halting_logits)
            q = compute_q_from_h(h)
            
            # === NaN Detection (AFTER q computation) ===
            if torch.isnan(q).any() or torch.isinf(q).any():
                if rank == 0:
                    xm.master_print(f"[Core {rank}] ⚠️  NaN/Inf in q at step {global_step}")
                    xm.master_print(f"  h range: [{h.min()}, {h.max()}]")
                    xm.master_print(f"  q range: [{q.min()}, {q.max()}]")
                raise ValueError("NaN detected in q probabilities")
            
            # Classification loss
            labels = teacher_label.float().unsqueeze(1).expand(-1, L)
            
            if class_logits.size(-1) == 2:
                class_logits_positive = class_logits[:, :, 1]
            else:
                class_logits_positive = class_logits.squeeze(-1)
            
            ce_per_layer = bce_loss_fn(class_logits_positive, labels)
            loss_cls = (q * ce_per_layer).sum(dim=1).mean()
            
            # Halting loss
            depths = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
            halt_penalty = (depths * (1 - h)).sum(dim=1)
            progress = global_step / total_steps
            lambda_now = lambda_start + (lambda_target - lambda_start) * progress
            loss_halt = lambda_now * halt_penalty.mean()
            
            # Total loss
            loss = loss_cls + loss_halt
            
            # === Final NaN Detection ===
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    xm.master_print(f"[Core {rank}] ⚠️  NaN/Inf in FINAL LOSS at step {global_step}")
                    xm.master_print(f"  loss_cls: {loss_cls}, loss_halt: {loss_halt}")
                raise ValueError("NaN detected in final loss")
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Reduced from 1.0
            xm.optimizer_step(optimizer)
            xm.mark_step()
            
            global_step += 1
        
        # --- End of Epoch Diagnostics ---
        loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
        loss_cls_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
        loss_halt_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
        h_mean = xm.all_reduce(xm.REDUCE_SUM, h.mean())
        
        # Weight sync check
        param_checks = []
        for i, param in enumerate(model.parameters()):
            if i >= 3:
                break
            param_element = param.flatten()[0]
            param_sum = xm.all_reduce(xm.REDUCE_SUM, param_element)
            param_checks.append((param_sum, param_element))
        
        xm.mark_step()
        
        if rank == 0:
            
            def format_diagnostic_output(logits_cpu, label_cpu, sample_type):
                if logits_cpu is None or label_cpu is None:
                    return f"  No {sample_type} sample found in the first batch of this core."

                sample_probs = torch.softmax(logits_cpu, dim=-1)
                predicted_classes = torch.argmax(sample_probs, dim=-1).tolist()
                max_confidences = sample_probs.max(dim=-1).values
                max_confidences_list = max_confidences.tolist()
                predicted_probs = [f"{p:.4f}" for p in max_confidences_list]
                true_label_value = label_cpu.item()
                
                output = []
                output.append(f"  --- {sample_type} Sample (True Label: {true_label_value}) ---")
                output.append(f"  Predicted Class per Layer (0-23): {predicted_classes}")
                output.append(f"  Max Confidence per Layer: {predicted_probs}")
                return "\n".join(output)
            
            elapsed = time.time() - start_time
            xm.master_print("-" * 80)
            xm.master_print(f"EPOCH {epoch+1}/{num_epochs} COMPLETED")
            xm.master_print(f"  Total Elapsed Time: {elapsed:.1f}s")
            xm.master_print("")
            xm.master_print("FINAL METRICS:")
            xm.master_print(f"  Step: {global_step}/{total_steps} ({global_step * 100 / total_steps:.1f}%)")
            xm.master_print(f"  Avg Total Loss: {loss_sum / num_cores}")
            xm.master_print(f"  Avg Cls Loss: {loss_cls_sum / num_cores}")
            xm.master_print(f"  Avg Halt Loss: {loss_halt_sum / num_cores}")
            xm.master_print(f"  Avg Mean h: {h_mean / num_cores}")
            xm.master_print(f"  Lambda: {lambda_start + (lambda_target - lambda_start) * (global_step / total_steps):.6f}")
            
            if sample_logits_cpu is not None and sample_label_cpu is not None:
                xm.master_print("\nDUAL SAMPLE CLASSIFICATION DIAGNOSTIC (One Positive & One Negative Sample):")
                pos_output = format_diagnostic_output(sample_logits_pos_cpu, sample_label_pos_cpu, "Positive (Label 1)")
                xm.master_print(pos_output)
                neg_output = format_diagnostic_output(sample_logits_neg_cpu, sample_label_neg_cpu, "Negative (Label 0)")
                xm.master_print(neg_output)
            
            xm.master_print("")
            xm.master_print("WEIGHT SYNCHRONIZATION CHECK:")
            for i, (param_sum, param_element) in enumerate(param_checks):
                expected_sum = param_element * num_cores
                diff = param_sum - expected_sum
                xm.master_print(f"  Parameter {i}: Difference = {diff}")
            xm.master_print("-" * 80)
        
        # --- Checkpointing ---
        if (epoch + 1) % flags["checkpoint_interval"] == 0:
            if rank == 0:
                checkpoint_path = f"/tmp/controller_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                xm.master_print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        xm.rendezvous(f"epoch_end_{epoch}")

    # --- FINAL SUMMARY ---
    total_time = time.time() - start_time
    if rank == 0:
        xm.master_print("\n" + "=" * 80)
        xm.master_print("TRAINING FINISHED")
        xm.master_print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        xm.master_print(f"Total steps: {global_step}")
        xm.master_print(f"✅ TRAINING COMPLETED ON ALL CORES. Active cores: {num_cores}/32")
        xm.master_print("=" * 80)
    
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
    FLAGS = {
        # Model architecture
        "d_ctrl": 256,
        "transformer_layers": 8,  # CHANGED: From 12 to 8
        
        # Optimization
        "lr": 1e-4,  # CHANGED: Reduced from 3e-4 for stability
        "batch_size": 64,
        
        # Halting loss schedule
        "lambda_start": 0.0001,
        "lambda_target": 0.025,
        
        # Training
        "epochs": 10,
        
        # Data loading
        "chunk_filename": "embeddings_chunk_0.npz",
        "samples_per_shard": 19500,
        
        # Logging and checkpointing
        "log_interval": 50,
        "checkpoint_interval": 1,
    }
    
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method='fork')