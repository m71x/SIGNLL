import os
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler # Required for splitting data

from controller_model import Controller
from training_data_download import training_data_download

# =========================================================================
# EVALUATION FUNCTION
# =========================================================================
def evaluate_model(rank, model, chunk_idx, threshold, batch_size, samples_per_shard):
    """
    All 32 cores download the same shard, but process separate parts.
    Results are combined using all_reduce.
    """
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    
    if rank == 0:
        xm.master_print(f"\n{'*'*80}")
        xm.master_print(f"*** STARTING DISTRIBUTED EVAL ON CHUNK {chunk_idx} (Threshold: {threshold}) ***")
        xm.master_print(f"{'*'*80}")

    model.eval()
    
    current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
    
    # Every core downloads the same file
    data = training_data_download(
        core_id=rank, 
        filename=current_chunk_filename,
        max_entries=samples_per_shard
    )
    
    if data is None:
        print(f"❌ [Core {rank}] Failed to load test data")
        return

    teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_label_full = torch.from_numpy(data['classifications']).long()
    
    if teacher_cls_full.shape[1] == 25:
        teacher_cls_full = teacher_cls_full[:, 1:25, :]
    
    dataset = TensorDataset(teacher_cls_full, teacher_label_full)
    
    # Use DistributedSampler to ensure each rank gets a separate 1/32 slice of the data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2
    )
    
    # Local counters (on device)
    local_correct = torch.tensor(0.0, device=device)
    local_samples = torch.tensor(0.0, device=device)
    local_exit_counts = torch.zeros(24, device=device)

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
            exit_indices[~row_has_exit] = 23 
            
            batch_indices = torch.arange(class_logits.size(0), device=device)
            selected_logits = class_logits[batch_indices, exit_indices]
            predictions = torch.argmax(selected_logits, dim=-1)
            
            # Update local tensors
            local_correct += (predictions == teacher_label).sum()
            local_samples += teacher_label.size(0)
            
            # Update local exit distribution
            # Efficiently increment exit counts on TPU
            ones = torch.ones_like(exit_indices, dtype=torch.float32)
            local_exit_counts.scatter_add_(0, exit_indices, ones)
            
            xm.mark_step()
            
            if rank == 0 and i % 50 == 0:
                print(f"[Eval] Core 0 processing batch {i}...")

    # --- COMBINE RESULTS FROM ALL 32 CORES ---
    # all_reduce sums the values across all cores in the TPU pod
    total_correct = xm.all_reduce(xm.REDUCE_SUM, local_correct)
    total_samples = xm.all_reduce(xm.REDUCE_SUM, local_samples)
    global_exit_counts = xm.all_reduce(xm.REDUCE_SUM, local_exit_counts)

    # Move to CPU for final print (Rank 0 only)
    if rank == 0:
        correct_val = total_correct.item()
        samples_val = total_samples.item()
        exit_counts_cpu = global_exit_counts.cpu()
        
        accuracy = (correct_val / samples_val) * 100.0
        layers = torch.arange(24, dtype=torch.float32)
        avg_exit_layer = (exit_counts_cpu * layers).sum() / samples_val
        std_exit_layer = torch.sqrt((exit_counts_cpu * (layers - avg_exit_layer).pow(2)).sum() / samples_val)

        xm.master_print(f"FINAL AGGREGATED RESULTS (32 CORES):")
        xm.master_print(f"  Accuracy: {accuracy:.2f}% ({int(correct_val)}/{int(samples_val)})")
        xm.master_print(f"  Avg Exit: {avg_exit_layer:.2f} +/- {std_exit_layer:.2f}")
        xm.master_print(f"  Distribution: {exit_counts_cpu.long().tolist()}")
        xm.master_print(f"{'*'*80}\n")

    # Final barrier to ensure all cores finish aggregation before moving to next threshold
    xm.rendezvous("threshold_complete")


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

    # 2. Loading Weights
    load_path = os.path.expanduser("~/SIGNLL/final_model_stage2.pt")
    if not os.path.exists(load_path):
        if rank == 0: print(f"❌ ERROR: Model not found at {load_path}")
        xm.rendezvous("abort")
        return

    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)
    xm.mark_step()
    
    if rank == 0:
        xm.master_print(f"✅ Loaded model. Starting evaluation on {xm.xrt_world_size()} cores.")

    xm.rendezvous("model_ready")

    # 3. Evaluation Loop
    test_chunk = flags.get("test_chunk", 29)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    for thresh in thresholds:
        evaluate_model(rank, model, test_chunk, thresh, flags["batch_size"], flags["samples_per_shard"])

    if rank == 0:
        xm.master_print("✅ Evaluation script finished successfully.")

def _mp_fn(rank, flags):
    try:
        torch.set_default_tensor_type('torch.FloatTensor')
        eval_main(rank, flags)
    except Exception as e:
        print(f" 🔥 [Core {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        os._exit(1) 

if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "batch_size": 32,
        "samples_per_shard": 39000, 
        "test_chunk": 29
    }  
    
    # Using 'fork' for faster startup on TPU VMs
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')