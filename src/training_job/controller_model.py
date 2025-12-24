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

from training_data_download import training_data_download

# =========================================================================
# UTILS
# =========================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        w12_out = self.w12(x)
        x1, x2 = w12_out.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.dropout(self.w3(hidden))

class CustomTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.nhead = nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)
        return x

# =========================================================================
# FIXED: EINSUM ENTROPY GATE (TPU STABLE)
# =========================================================================
class VectorizedEntropyGate(nn.Module):
    def __init__(self, d_ctrl: int, L: int):
        super().__init__()
        # REPLACEMENT: Instead of Conv1d(groups=L), we use explicit parameters
        # and torch.einsum. This is native to TPU systolic arrays and stable.
        self.L = L
        self.d_ctrl = d_ctrl
        
        # Weights for Layer 1: [L, D, 8] 
        # (L independent matrices of D->8)
        self.w1 = nn.Parameter(torch.empty(L, d_ctrl, 8))
        self.b1 = nn.Parameter(torch.zeros(L, 8))
        
        # Weights for Layer 2: [L, 8, 1]
        self.w2 = nn.Parameter(torch.empty(L, 8, 1))
        self.b2 = nn.Parameter(torch.zeros(L, 1))
        
        self.act = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Init with small values for stability
        nn.init.normal_(self.w1, mean=0.0, std=0.01)
        nn.init.normal_(self.w2, mean=0.0, std=0.01)

    def forward(self, z, entropy):
        # z: [B, L, D]
        # w1: [L, D, 8]
        # b1: [L, 8]
        
        # 1. First Projection (D -> 8, per layer)
        # 'bld,ldh->blh' means:
        #   For each item in Batch (b), and each Layer (l):
        #   Dot product feature vector (d) with that Layer's weight matrix (dh)
        h = torch.einsum('bld,ldh->blh', z, self.w1) + self.b1
        h = self.act(h)
        
        # 2. Second Projection (8 -> 1, per layer)
        out = torch.einsum('blh,lho->blo', h, self.w2) + self.b2
        
        # 3. Gating
        gate = self.sigmoid(out) # [B, L, 1]
        
        return entropy * gate

# =========================================================================
# CONTROLLER (Re-defined with new Gate)
# =========================================================================
class Controller(nn.Module):
    def __init__(self, L=24, d_teacher=1024, d_ctrl=256, n_layers=12, n_heads=4, ffn_dim=1024, dropout=0.1, halting_bias_init=-4.0, num_classes=2):
        super().__init__()
        self.L = L
        self.d_teacher = d_teacher
        self.d_ctrl = d_ctrl
        self.num_classes = num_classes

        self.input_ln = nn.LayerNorm(d_teacher)
        self.proj = nn.Linear(d_teacher, d_ctrl)
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_ctrl, 2).float() / d_ctrl))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(L, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) 
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_ctrl, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.pre_ln = nn.LayerNorm(d_ctrl)
        self.post_ln = nn.LayerNorm(d_ctrl) 

        # NEW EINSUM GATE
        self.entropy_gate_module = VectorizedEntropyGate(d_ctrl, L)
        
        self.halting_heads = nn.ModuleList([
            nn.Linear(d_ctrl + 1, 1) for _ in range(L)
        ])
        self.classifier_heads = nn.ModuleList([
            nn.Linear(d_ctrl, self.num_classes) for _ in range(L)
        ])

        self.apply(self._init_weights)
        with torch.no_grad():
            for head in self.halting_heads:
                head.bias.fill_(halting_bias_init)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, teacher_cls):
        import math
        B, L, D = teacher_cls.shape
        x = self.input_ln(teacher_cls)
        x = self.proj(x)
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        x = self.pre_ln(x)
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        z = x
        for layer in self.layers:
            z = layer(z, mask=attn_mask)
        z = self.post_ln(z) 
        
        class_logits_list = []
        entropy_list = []
        max_entropy = math.log(self.num_classes)
        
        for l in range(L):
            c_l = self.classifier_heads[l](z[:, l, :]) 
            class_logits_list.append(c_l)
            log_probs = F.log_softmax(c_l, dim=-1)
            probs = torch.exp(log_probs)
            probs_safe = probs.clamp(min=1e-10, max=1.0)
            log_probs_safe = torch.log(probs_safe)
            ent = -(probs_safe * log_probs_safe).sum(dim=-1, keepdim=True)
            ent_norm = (ent / max_entropy).clamp(0.0, 1.0)
            entropy_list.append(ent_norm)
            
        all_entropies = torch.stack(entropy_list, dim=1)
        
        # New Stable Gate Forward
        all_gated_entropies = self.entropy_gate_module(z, all_entropies)

        halting_logits_list = []
        for l in range(L):
            gated_ent_l = all_gated_entropies[:, l, :]
            combined_input = torch.cat([z[:, l, :], gated_ent_l], dim=-1)
            h_l = self.halting_heads[l](combined_input) 
            halting_logits_list.append(h_l)
        
        halting_logits = torch.cat(halting_logits_list, dim=-1) 
        class_logits = torch.stack(class_logits_list, dim=1)
        return halting_logits, class_logits, z

# =========================================================================
# TRAINING LOOP 
# =========================================================================
def train_loop(rank, flags):
    device = xm.xla_device()
    num_cores = xm.xrt_world_size()
    
    L = 24
    model = Controller(
        L=L,
        d_teacher=1024,
        d_ctrl=flags["d_ctrl"],
        n_layers=flags["transformer_layers"],
        num_classes=2
    ).to(device)
    
    if rank == 0: xm.master_print("Synchronizing initial weights...")
    for param in model.parameters():
        param.data = xm.all_reduce(xm.REDUCE_SUM, param.data) / num_cores
    xm.mark_step()
    xm.rendezvous("weights_synced") 

    for stage in [2]: 
        if stage == 1:
            for param in model.parameters(): param.requires_grad = True
            for param in model.halting_heads.parameters(): param.requires_grad = False
            for param in model.entropy_gate_module.parameters(): param.requires_grad = False
        elif stage == 2:
            for param in model.parameters(): param.requires_grad = False
            for param in model.halting_heads.parameters(): param.requires_grad = True
            for param in model.entropy_gate_module.parameters(): param.requires_grad = True

        if rank == 0: xm.master_print(f"STARTING STAGE {stage}")
        
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(model_params, lr=flags["lr"], weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-6)

        for chunk_idx in range(28): 
            current_chunk_filename = f"embeddings_chunk_{chunk_idx}.npz"
            if rank == 0: xm.master_print(f"Loading {current_chunk_filename}")

            data = training_data_download(
                core_id=rank,
                filename=current_chunk_filename,
                max_entries=flags["samples_per_shard"]
            )
            
            if data is None: raise RuntimeError(f"[Core {rank}] Failed load")

            teacher_cls_full = torch.from_numpy(data['all_layer_cls_tokens']).float()
            teacher_label_full = torch.from_numpy(data['classifications']).long()
            if 'teacher_logits' in data:
                t_logits = torch.from_numpy(data['teacher_logits']).float()
                teacher_log_probs_full = F.log_softmax(t_logits / 2.0, dim=-1)
            else:
                teacher_log_probs_full = torch.zeros(teacher_label_full.size(0), 2) # Dummy

            if teacher_cls_full.shape[1] == 25: teacher_cls_full = teacher_cls_full[:, 1:25, :]
            
            N_total_local = teacher_cls_full.shape[0]
            N_target = (N_total_local // num_cores) * 32
            teacher_cls_full = teacher_cls_full[:N_target]
            teacher_label_full = teacher_label_full[:N_target]
            teacher_log_probs_full = teacher_log_probs_full[:N_target] 

            neg_samples = (teacher_label_full == 0).sum().item()
            pos_samples = (teacher_label_full == 1).sum().item()
            pos_weight_val = neg_samples / (pos_samples + 1e-6)
            pos_weight_tensor = torch.tensor([pos_weight_val]).float().to(device)

            dataset = TensorDataset(teacher_cls_full, teacher_label_full, teacher_log_probs_full)
            sampler = RandomSampler(dataset)
            data_loader = DataLoader(dataset, sampler=sampler, batch_size=flags["batch_size"], drop_last=True, num_workers=2)
            
            for epoch in range(flags["epochs"]):
                model.train()
                for batch_idx, (teacher_cls, teacher_label, _) in enumerate(data_loader):
                    teacher_cls = teacher_cls.to(device)
                    teacher_label = teacher_label.to(device)
                    
                    if stage == 2:
                        halting_logits, class_logits, z = model(teacher_cls)
                        predictions = torch.argmax(class_logits, dim=-1)
                        is_correct = (predictions == teacher_label.unsqueeze(1)).float()
                        
                        # STATIC WEIGHTS for Stability Check
                        sample_weights = torch.ones_like(halting_logits)

                        loss_halt = F.binary_cross_entropy_with_logits(
                            halting_logits, is_correct, weight=sample_weights
                        )
                        
                        h = torch.sigmoid(halting_logits)
                        h_safe = h.clamp(min=1e-4, max=1.0 - 1e-4)
                        h_entropy = -(h_safe * h_safe.log() + (1 - h_safe) * (1 - h_safe).log())
                        loss_entropy = -0.0025 * h_entropy.mean()
                        
                        loss = loss_halt + loss_entropy
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        xm.optimizer_step(optimizer)
                        scheduler.step()
                        xm.mark_step()

                        if rank == 0 and batch_idx == 0:
                            xm.mark_step()
                            with torch.no_grad():
                                d_halt, _, _ = model(teacher_cls)
                                probs = torch.sigmoid(d_halt[0]).detach().cpu().tolist()
                                xm.master_print(f"  DIAG [0]: {[f'{p:.2f}' for p in probs]}")

                loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss)
                xm.rendezvous(f"ep_end_{epoch}")
                if rank == 0: xm.master_print(f"  Loss: {loss_sum / num_cores:.4f}")

    xm.rendezvous("done")

def _mp_fn(rank, flags):
    try:
        torch.set_default_tensor_type('torch.FloatTensor')
        train_loop(rank, flags)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    BASE_FLAGS = {
        "d_ctrl": 512, "transformer_layers": 4, "lr": 5e-4, "batch_size": 64, "epochs": 5, "samples_per_shard": 39000
    }
    xmp.spawn(_mp_fn, args=(BASE_FLAGS,), start_method='fork')