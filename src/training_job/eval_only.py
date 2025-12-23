import os
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from controller_model import Controller
from training_data_download import training_data_download

def evaluate_model(rank, model, threshold, batch_size, samples_per_shard):
    """
    Every core downloads Core 0's shard, but splits the 20k samples 
    equally among all 32 cores.
    """
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    
    if rank == 0:
        xm.master_print(f"\n{'='*80}")
        xm.master_print(f"EVALUATING: Core 0's Chunk 29 | Threshold: {threshold}")
        xm.master_print(f"{'='*80}")

    model.eval()
    
    # FORCE every core to download Core 0's data
    data = training_data_download(
        core_id=0, # Fixed to 0 so everyone gets the same shard
        filename="embeddings_chunk_29.npz",
        max_entries=samples_per_shard
    )
    
    if data is None:
        print(f"❌ [Rank {rank}] Failed to load shared data.")
        return

    # Convert to Tensors
    teacher_cls = torch.from_numpy(data['all_layer_cls_tokens']).float()
    teacher_labels = torch.from_numpy(data['classifications']).long()
    
    if teacher_cls.shape[1] == 25:
        teacher_cls = teacher_cls[:, 1:25, :]
    
    dataset = TensorDataset(teacher_cls, teacher_labels)
    
    # DistributedSampler splits Core 0's 20k samples into 32 equal parts
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    local_correct = torch.tensor(0.0, device=device)
    local_samples = torch.tensor(0.0, device=device)
    local_exit_counts = torch.zeros(24, device=device)

    with torch.no_grad():
        for teacher_cls_batch, labels_batch in data_loader:
            teacher_cls_batch = teacher_cls_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            halting_logits, class_logits, _ = model(teacher_cls_batch)
            h_probs = torch.sigmoid(halting_logits)
            
            # Logic for Early Exit
            threshold_mask = (h_probs > threshold)
            exit_indices = torch.argmax(threshold_mask.long(), dim=1)
            row_has_exit = threshold_mask.any(dim=1)
            exit_indices[~row_has_exit] = 23 
            
            batch_indices = torch.arange(class_logits.size(0), device=device)
            selected_logits = class_logits[batch_indices, exit_indices]
            predictions = torch.argmax(selected_logits, dim=-1)
            
            # Local Accumulation
            local_correct += (predictions == labels_batch).sum()
            local_samples += labels_batch.size(0)
            
            # Exit Distribution
            ones = torch.ones_like(exit_indices, dtype=torch.float32)
            local_exit_counts.scatter_add_(0, exit_indices, ones)
            
            xm.mark_step()

    # Aggregate results from all 32 cores
    total_correct = xm.all_reduce(xm.REDUCE_SUM, local_correct)
    total_samples = xm.all_reduce(xm.REDUCE_SUM, local_samples)
    global_exit_counts = xm.all_reduce(xm.REDUCE_SUM, local_exit_counts)

    if rank == 0:
        acc = (total_correct.item() / total_samples.item()) * 100
        dist = global_exit_counts.cpu().long().tolist()
        xm.master_print(f"DONE | Accuracy: {acc:.2f}% | Samples: {int(total_samples.item())}")
        xm.master_print(f"Exit Distribution: {dist}\n")

    xm.rendezvous("threshold_step")

def eval_main(rank, flags):
    device = xm.xla_device()
    
    model = Controller(
        L=24, d_teacher=1024, d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"], num_classes=2
    ).to(device)

    if rank == 0:
    # Load Stage 2 Weights
        load_path = os.path.expanduser("~/SIGNLL/final_model_stage2_gated.pt")
        state_dict = torch.load(load_path, map_location='cpu')
        model.load_state_dict(state_dict)
        xm.mark_step()

        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for thresh in thresholds:
            evaluate_model(rank, model, thresh, flags["batch_size"], flags["samples_per_shard"])

if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512,
        "transformer_layers": 4,
        "batch_size": 32,
        "samples_per_shard": 39000 # Max limit to look for in chunk 29
    }
    xmp.spawn(eval_main, args=(BASE_FLAGS,), start_method='fork')