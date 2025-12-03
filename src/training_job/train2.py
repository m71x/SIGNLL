import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from controller_model import Controller, compute_q_from_h
from training_data_download import training_data_download

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
# TRAINING LOOP (Modified for Data Slicing)
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
        
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_optimize, lr=flags["lr"], weight_decay=1e-2)

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
            
            if teacher_cls_full.shape[1] == 25:
                teacher_cls_full = teacher_cls_full[:, 1:25, :]
            
            # --- START MODIFICATION: Data Slicing to 12.5% (4/32) ---
            N_total_local = teacher_cls_full.shape[0]
            # Calculate 1/32 of the local data (original usage) and multiply by 4
            N_target = (N_total_local // num_cores) * 4 

            # Apply the slice
            teacher_cls_full = teacher_cls_full[:N_target]
            teacher_label_full = teacher_label_full[:N_target]

            if rank == 0:
                xm.master_print(f"Data Sliced: Using {N_target}/{N_total_local} samples ({N_target/N_total_local:.2%}) for 12.50% utilization.")
            # --- END MODIFICATION ---

            # Class Weighting
            neg_samples = (teacher_label_full == 0).sum().item()
            pos_samples = (teacher_label_full == 1).sum().item()
            pos_weight_val = neg_samples / (pos_samples + 1e-6)
            pos_weight_tensor = torch.tensor([pos_weight_val]).float().to(device)

            bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor).to(device)

            dataset = TensorDataset(teacher_cls_full, teacher_label_full)
            
            # Use RandomSampler to shuffle the reduced dataset (essential fix)
            sampler = RandomSampler(dataset)
            
            data_loader = DataLoader(dataset, sampler=sampler, batch_size=flags["batch_size"], drop_last=True, num_workers=2)
            
            # --- Epoch Loop ---
            for epoch in range(flags["epochs"]):
                model.train()
                
                # Reset diagnostic holders for this epoch
                diag_sample_pos = None
                diag_sample_neg = None

                for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                    if stage == 2: global_step += 29 
                    
                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    
                    # Forward Pass
                    halting_logits, class_logits, _ = model(teacher_cls)
                    
                    # --- Capture Diagnostics (Rank 0, First Batch Only) ---
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

                    # --- LOSS CALCULATION ---
                    labels = teacher_label.float().unsqueeze(1).expand(-1, L)
                    if class_logits.size(-1) == 2:
                        class_logits_positive = class_logits[:, :, 1]
                    else:
                        class_logits_positive = class_logits.squeeze(-1)
                    
                    ce_per_layer = bce_loss_fn(class_logits_positive, labels)

                    if stage == 1:
                        loss_cls = ce_per_layer.mean()
                        loss_halt = torch.tensor(0.0, device=device)
                        loss = loss_cls
                        h = torch.zeros_like(halting_logits) 

                    elif stage == 2:
                        h = torch.sigmoid(halting_logits)
                        q = compute_q_from_h(h)
                        loss_cls = (q * ce_per_layer).sum(dim=1).mean()
                        
                        # Linear Depth Penalty
                        depths = (torch.arange(1, L + 1, device=device).float()).unsqueeze(0)
                        halt_penalty = (depths * (1 - h)).sum(dim=1)
                        
                        progress = global_step / total_steps_stage_2
                        lambda_now = flags["lambda_start"] + (flags["lambda_target"] - flags["lambda_start"]) * progress
                        loss_halt = lambda_now * halt_penalty.mean()
                        
                        loss = loss_cls + loss_halt

                    # Optimization
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    xm.optimizer_step(optimizer)
                    xm.mark_step()

                # --- End of Epoch Aggregation ---
                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                loss_cls_sum = xm.all_reduce(xm.REDUCE_SUM, loss_cls)
                loss_halt_sum = xm.all_reduce(xm.REDUCE_SUM, loss_halt)
                
                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")
                
                # --- Rank 0 Logging ---
                if rank == 0:
                    elapsed = time.time() - start_time
                    xm.master_print("-" * 60)
                    xm.master_print(f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}")
                    xm.master_print(f"  Total Loss: {loss_sum / num_cores:.4f}")
                    xm.master_print(f"  Cls Loss:   {loss_cls_sum / num_cores:.4f}")
                    xm.master_print(f"  Halt Loss:  {loss_halt_sum / num_cores:.4f}")
                    
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

            # Checkpoint
            if (chunk_idx + 1) % 5 == 0 and rank == 0:
                torch.save(model.state_dict(), f"/tmp/controller_stage{stage}_chunk{chunk_idx+1}.pt")
                xm.master_print("Saved Checkpoint.")

            xm.rendezvous(f"chunk_end_st{stage}_ch{chunk_idx}")


    # Final Save and Evaluation
    if rank == 0:
        torch.save(model.state_dict(), "/tmp/controller_final_stage2.pt")
        xm.master_print("✅ Training Complete.")

    test_chunk = flags.get("test_chunk", 29) 
    evaluate_model(rank, model, test_chunk, 0.5, flags["batch_size"], flags["samples_per_shard"])
    evaluate_model(rank, model, test_chunk, 0.6, flags["batch_size"], flags["samples_per_shard"])
    evaluate_model(rank, model, test_chunk, 0.7, flags["batch_size"], flags["samples_per_shard"])
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
        "lr": 1e-4,
        "batch_size": 32, 
        "lambda_start": 0.0001,
        "lambda_target": 0.004,
        "epochs": 4,
        "samples_per_shard": 39000, 
        "test_chunk": 29, 
        "test_threshold": 0.8
    }
    
    print("Starting Two-Stage XLA Job.")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')