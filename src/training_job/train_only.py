import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
# from training_data_download import training_data_download # Not needed for dummy test

from controller_model import Controller

# =========================================================================
# TRAINING LOOP (DUMMY DATA & STAGE 2 ONLY)
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

    # =========================================================================
    # STAGE LOOP: SKIPPING 1, RUNNING 2
    # =========================================================================
    # We only run Stage 2 for this test
    for stage in [2]:
        
        # --- PHASE SETUP ---
        stage_name = "STAGE 2 TEST: Halting Heads + Entropy Gates (Backbone FROZEN)"
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
            xm.master_print(f"STARTING {stage_name} (DUMMY DATA)")
            xm.master_print(f"{'#'*80}")
        
        # --- OPTIMIZER SETUP ---
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(model_params, lr=flags["lr"], weight_decay=1e-2)

        # --- SCHEDULER SETUP ---
        # Dummy scheduler for test
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

        # Run 5 "Chunks" of dummy data
        for chunk_idx in range(5): 
            if rank == 0:
                xm.master_print(f"Stage {stage} | Chunk {chunk_idx + 1}/5 | Generatng Dummy Data...")

            # --- DUMMY DATA GENERATION ---
            # Simulate: [Batch*Steps, L, D] -> We just make enough for 10 batches
            steps_per_chunk = 10
            N_dummy = flags["batch_size"] * steps_per_chunk
            
            # [N, L=24, D=1024]
            teacher_cls_full = torch.randn(N_dummy, 24, 1024)
            # [N] Random labels 0 or 1
            teacher_label_full = torch.randint(0, 2, (N_dummy,)).long()
            
            # Pos Weight (Static)
            pos_weight_tensor = torch.tensor([1.0]).float().to(device)
            
            dataset = TensorDataset(teacher_cls_full, teacher_label_full)
            sampler = RandomSampler(dataset)
            
            data_loader = DataLoader(
                dataset, sampler=sampler, batch_size=flags["batch_size"], 
                drop_last=True, num_workers=0 # No workers for dummy
            )
            
            # --- Epoch Loop (Just 1 epoch for test) ---
            for epoch in range(1):
                model.train()
                diag_sample_pos = None
                diag_sample_neg = None

                for batch_idx, (teacher_cls, teacher_label) in enumerate(data_loader):
                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    
                    # --- STAGE 2 LOGIC ---
                    # Forward pass
                    halting_logits, class_logits, z = model(teacher_cls)
                    
                    predictions = torch.argmax(class_logits, dim=-1)
                    is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                    
                    # CLASS-AWARE REBALANCING
                    n_pos = (teacher_label == 1).sum().float()
                    n_neg = (teacher_label == 0).sum().float()
                    neg_weight_val = (n_pos / (n_neg + 1e-6)).clamp(min=1.0)
                    
                    sample_weights = torch.ones_like(halting_logits)
                    sample_weights[teacher_label == 0] = neg_weight_val.item()
                    
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
                    
                    # --- FIX: DIAGNOSTICS FOR STAGE 2 ---
                    # Move to CPU on ALL ranks to preserve SPMD graph consistency
                    if batch_idx == 0:
                        c_logits_cpu = class_logits.detach().cpu()
                        h_logits_cpu = halting_logits.detach().cpu()
                        lbl_cpu = teacher_label.detach().cpu()

                        if rank == 0:
                            def extract_sample(label_val):
                                # Safe to use nonzero() on CPU tensor
                                indices = (lbl_cpu == label_val).nonzero(as_tuple=True)[0]
                                if indices.numel() > 0:
                                    idx = indices[0].item()
                                    return {
                                        'cls': c_logits_cpu[idx],
                                        'halt': h_logits_cpu[idx],
                                        'lbl': lbl_cpu[idx]
                                    }
                                return None
                            
                            diag_sample_pos = extract_sample(1)
                            diag_sample_neg = extract_sample(0)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    xm.mark_step()

                # --- END OF CHUNK LOGGING ---
                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                loss_log = loss_sum 
                
                xm.rendezvous(f"ep_end_st{stage}_ch{chunk_idx}_ep{epoch}")
                
                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    xm.master_print("-" * 60)
                    xm.master_print(f"STAGE {stage} | CHUNK {chunk_idx+1} | EPOCH {epoch+1}")
                    xm.master_print(f"  Total Loss: {loss_sum / num_cores:.4f}")
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
    if rank == 0:
        xm.master_print("✅ TEST PASSED: Stage 2 completed without hang.")
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
        "epochs": 1,
        "samples_per_shard": 1000 # Dummy count
    }  
    
    print("Starting Dummy Stage 2 Test.")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')