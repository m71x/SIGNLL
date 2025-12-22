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
    Runs ONLY on Device 0 logic, but workers must be handled.
    """
    # Note: In XLA, even if only Rank 0 does the math, we must ensure 
    # no global syncs are triggered inside here that workers aren't part of.
    if rank != 0:
        return

    device = xm.xla_device()
    
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
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2
    )
    xm.master_print("data loader set up")
    total_samples = 0
    total_correct = 0
    layer_exit_counts_cpu = torch.zeros(24, dtype=torch.float32)
    xm.master_print("going into loop")
    with torch.no_grad():
        for i, (teacher_cls, teacher_label) in enumerate(data_loader):
            teacher_cls = teacher_cls.to(device)
            teacher_label = teacher_label.to(device)
            xm.master_print("teacher sample set up")
            halting_logits, class_logits, _ = model(teacher_cls)
            xm.master_print("inference performedp")
            h_probs = torch.sigmoid(halting_logits)
            threshold_mask = (h_probs > threshold)
            
            # Determine exit layer
            exit_indices = torch.argmax(threshold_mask.long(), dim=1)
            row_has_exit = threshold_mask.any(dim=1)
            exit_indices[~row_has_exit] = 23 # Default to last layer
            xm.master_print("exit layer determined")
            batch_indices = torch.arange(class_logits.size(0), device=device)
            selected_logits = class_logits[batch_indices, exit_indices]
            predictions = torch.argmax(selected_logits, dim=-1)
            xm.master_print("p2")

            correct_tensor = (predictions == teacher_label).sum()
            total_correct += correct_tensor.item() 
            total_samples += teacher_label.size(0)
            xm.master_print("correctness calculated")
            # Track statistics on CPU
            exit_indices_cpu = exit_indices.cpu()
            unique_exits, counts = torch.unique(exit_indices_cpu, return_counts=True)
            layer_exit_counts_cpu.index_add_(0, unique_exits, counts.float())
            xm.master_print("statistics calculated")
            xm.master_print(i)
            # Local sync for Rank 0 to keep memory clean
            xm.mark_step()
            
            if i % 100 == 0:
                print(f"[Eval] Processed batch {i}...")
            
    xm.master_print("loop exited")
    xm.mark_step()
    accuracy = (total_correct / total_samples) * 100.0
    xm.mark_step()
    xm.master_print("accuracy calculated")
    # --- Stats ---
    layers = torch.arange(24, dtype=torch.float32)
    xm.master_print("layers arranged")
    xm.mark_step()

    avg_exit_layer = (layer_exit_counts_cpu * layers).sum() / total_samples
    xm.master_print("avg exit layer determined")
    xm.mark_step()

    std_exit_layer = torch.sqrt((layer_exit_counts_cpu * (layers - avg_exit_layer).pow(2)).sum() / total_samples)
    xm.master_print("done")
    xm.mark_step()



    xm.master_print(f"RESULTS:")
    xm.master_print(f"  Accuracy: {accuracy:.2f}%")
    xm.master_print(f"  Avg Exit: {avg_exit_layer:.2f} +/- {std_exit_layer:.2f}")
    xm.master_print(f"  Distribution: {layer_exit_counts_cpu.long().tolist()}")
    xm.master_print(f"{'*'*80}\n")
    xm.mark_step()


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
            print(f"❌ ERROR: Model not found at {load_path}")
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

    # 3. Evaluation
    test_chunk = flags.get("test_chunk", 29)
    thresholds = [0.95]
    
    
        # Rank 0 performs the actual evaluation
    for thresh in thresholds:
        evaluate_model(rank, model, test_chunk, thresh, flags["batch_size"], flags["samples_per_shard"])
        
        # After finishing all loops, Rank 0 signals workers to release
    xm.rendezvous("final")

    if rank == 0:
        xm.master_print("✅ Evaluation script finished successfully.")

def _mp_fn(rank, flags):
    try:
        # Ensure default tensors are on CPU for the initial load
        torch.set_default_tensor_type('torch.FloatTensor')
        eval_main(rank, flags)
    except Exception as e:
        print(f" 🔥 [Core {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # If one core fails, we try to force an exit to prevent hanging others
        os._exit(1) 

if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "batch_size": 32,
        "samples_per_shard": 39000, 
        "test_chunk": 29
    }  
    
    print("Starting Synchronized Evaluation Job.")
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')