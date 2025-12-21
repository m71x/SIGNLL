import os
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from controller_model import Controller
from training_data_download import training_data_download

# =========================================================================
# EVALUATION FUNCTION
# =========================================================================
def evaluate_model(rank, model, chunk_idx, threshold, batch_size, samples_per_shard):
    """
    Tests the model on a specific chunk using an early-exit strategy.
    Runs ONLY on Device 0 logic, but ALL ranks must call this function.
    """
    device = xm.xla_device()
    
    # Only rank 0 does the actual evaluation work
    if rank == 0:
        xm.master_print(f"\n{'*'*80}")
        xm.master_print(f"*** STARTING EVALUATION ON CHUNK {chunk_idx} (Threshold: {threshold}) ***")
        xm.master_print(f"{'*'*80}")

        model.eval()
        
        current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
        data = training_data_download(
            core_id=0, 
            filename=current_chunk_filename,
            max_entries=samples_per_shard
        )
        
        if data is None:
            xm.master_print(f"❌ [Core {rank}] Failed to load test data for chunk {chunk_idx}")
            return

        teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
        teacher_label_full = torch.from_numpy(data['classifications']).long()
        
        if teacher_cls_full.shape[1] == 25:
            teacher_cls_full = teacher_cls_full[:, 1:25, :]
        
        dataset = TensorDataset(teacher_cls_full, teacher_label_full)
        sampler = SequentialSampler(dataset)
        
        # CRITICAL FIX: Set num_workers=0 to avoid DataLoader deadlock
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            num_workers=0  # Changed from 2 to 0
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
                
                # Determine exit layer
                exit_indices = torch.argmax(threshold_mask.long(), dim=1)
                row_has_exit = threshold_mask.any(dim=1)
                exit_indices[~row_has_exit] = 23 # Default to last layer
                
                batch_indices = torch.arange(class_logits.size(0), device=device)
                selected_logits = class_logits[batch_indices, exit_indices]
                predictions = torch.argmax(selected_logits, dim=-1)
                
                correct_tensor = (predictions == teacher_label).sum()
                total_correct += correct_tensor.item() 
                total_samples += teacher_label.size(0)
                
                # Track statistics on CPU
                exit_indices_cpu = exit_indices.cpu()
                unique_exits, counts = torch.unique(exit_indices_cpu, return_counts=True)
                layer_exit_counts_cpu.index_add_(0, unique_exits, counts.float())
                
                if i % 100 == 0:
                    xm.master_print(f"[Eval] Processed batch {i}...")
                
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
        
        # --- Histogram Log ---
        xm.master_print(f"  Exit Layer Distribution (0-23): {layer_exit_counts_cpu.long().tolist()}")
        
        xm.master_print(f"{'*'*80}\n")

        model.train()
    
    # CRITICAL: All ranks must synchronize after evaluation to prevent XLA deadlock
    xm.mark_step()
    xm.rendezvous(f"eval_done_threshold_{threshold}") 

# =========================================================================
# EVALUATION LOOP
# =========================================================================
def eval_main(rank, flags):
    device = xm.xla_device()
    
    # 1. Initialize Model
    model = Controller(
        L=24,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)

    # 2. Loading Weights (All Ranks participate to avoid rendezvous mismatch)
    load_path = os.path.expanduser("~/SIGNLL/final_model_stage2.pt")
    
    # Critical: Check path existence on all ranks before trying to load
    if not os.path.exists(load_path):
        if rank == 0:
            xm.master_print(f"❌ ERROR: Model not found at {load_path}")
        # All ranks must hit this rendezvous before exiting
        xm.rendezvous("model_not_found_abort")
        return

    # All ranks load the state dict to CPU
    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Sync weights to TPU device and clear graph
    xm.mark_step()
    
    if rank == 0:
        xm.master_print(f"✅ Successfully loaded and synced model from {load_path}")

    # Everyone must reach this point
    xm.rendezvous("model_loaded_and_synced")

    # 3. Evaluation - ALL ranks must iterate through thresholds together
    test_chunk = flags.get("test_chunk", 29)
    thresholds = flags.get("thresholds", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    
    if rank == 0:
        xm.master_print(f"\n{'='*80}")
        xm.master_print(f"STARTING EVALUATION ON CHUNK {test_chunk}")
        xm.master_print(f"Testing thresholds: {thresholds}")
        xm.master_print(f"{'='*80}\n")
    
    # CRITICAL: All ranks must iterate through thresholds together
    # This prevents XLA synchronization deadlocks
    for threshold in thresholds:
        evaluate_model(
            rank, 
            model, 
            test_chunk, 
            threshold, 
            flags["batch_size"], 
            flags["samples_per_shard"]
        )
        # Rendezvous is now inside evaluate_model
    
    # Final synchronization
    xm.rendezvous("evaluation_complete")
    
    if rank == 0:
        xm.master_print("✅ Evaluation script finished successfully.")

def _mp_fn(rank, flags):
    try:
        # Ensure default tensors are on CPU for the initial load
        torch.set_default_tensor_type('torch.FloatTensor')
        eval_main(rank, flags)
    except Exception as e:
        print(f"🔥 [Core {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # If one core fails, we try to force an exit to prevent hanging others
        os._exit(1) 

if __name__ == "__main__":
    EVAL_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "batch_size": 32,
        "samples_per_shard": 39000,
        "test_chunk": 29,
        "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    }
    
    print("Starting Synchronized Evaluation Job.")
    xmp.spawn(_mp_fn, args=(EVAL_FLAGS,), start_method='fork')