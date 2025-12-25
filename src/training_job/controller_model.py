import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# =========================================================================
# RoPE Utility Functions
# =========================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

# =========================================================================
# SwiGLU & Transformer Layers
# =========================================================================
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
# NEW: TPU-Friendly Batch Linear Layer
# =========================================================================
class BatchLinear(nn.Module):
    """
    Applies a linear transformation to each element in the sequence L 
    using a DIFFERENT weight matrix for each step l in L.
    Input:  [Batch, L, D_in]
    Weight: [L, D_in, D_out]
    Output: [Batch, L, D_out]
    """
    def __init__(self, L: int, in_features: int, out_features: int):
        super().__init__()
        self.L = L
        self.in_features = in_features
        self.out_features = out_features
        
        # Shape: [L, In, Out]
        self.weight = nn.Parameter(torch.Tensor(L, in_features, out_features))
        # Shape: [L, Out]
        self.bias = nn.Parameter(torch.Tensor(L, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming init adapted for the batch dimension
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: [B, L, D_in]
        # Uses einsum for efficient batch matrix multiplication
        # b=batch, l=layers, i=input_dim, o=output_dim
        return torch.einsum('bli,lio->blo', x, self.weight) + self.bias

# =========================================================================
# OPTIMIZED: Vectorized Gating Module
# =========================================================================
class VectorizedEntropyGate(nn.Module):
    def __init__(self, d_ctrl: int, L: int):
        super().__init__()
        self.L = L
        
        # Layer 1: d_ctrl -> 8 (Unique weights per layer L)
        self.net1 = BatchLinear(L, d_ctrl, 8)
        self.act = nn.Tanh()
        
        # Layer 2: 8 -> 1 (Unique weights per layer L)
        self.net2 = BatchLinear(L, 8, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize to output 0.5 (bias=0)
        nn.init.constant_(self.net2.bias, 0.0)

    def forward(self, z, entropy):
        # z: [B, L, D]
        out = self.net1(z)      
        out = self.act(out)
        out = self.net2(out)             
        gate = self.sigmoid(out)
        return entropy * gate

# =========================================================================
# NEW: Vectorized Halting Head
# =========================================================================
class VectorizedHaltingHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, L: int):
        super().__init__()
        self.L = L
        
        # Layer 1: Input -> Hidden (Unique weights per layer L)
        self.net1 = BatchLinear(L, input_dim, hidden_dim)
        self.act = nn.ReLU()
        
        # Layer 2: Hidden -> 1 (Unique weights per layer L)
        self.net2 = BatchLinear(L, hidden_dim, 1)

    def forward(self, x):
        # x shape: [B, L, input_dim]
        out = self.net1(x)
        out = self.act(out)
        out = self.net2(out)
        
        # Output is [B, L, 1], squeeze to [B, L]
        return out.squeeze(-1)

# =========================================================================
# CONTROLLER
# =========================================================================
class Controller(nn.Module):
    def __init__(
        self,
        L: int = 24,
        d_teacher: int = 1024,
        d_ctrl: int = 256, 
        n_layers: int = 12,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        halting_bias_init: float = -4.0,
        num_classes: int = 2,
        halting_hidden_dim: int = None, 
    ):
        super().__init__()
        self.L = L
        self.d_teacher = d_teacher
        self.d_ctrl = d_ctrl
        self.num_classes = num_classes
        
        if halting_hidden_dim is None:
            halting_hidden_dim = d_ctrl * 2

        # 1. Inputs and Embeddings
        self.input_ln = nn.LayerNorm(d_teacher)
        self.proj = nn.Linear(d_teacher, d_ctrl)
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_ctrl, 2).float() / d_ctrl))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(L, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) 
        
        cos_cached = emb.cos().unsqueeze(0) 
        sin_cached = emb.sin().unsqueeze(0) 
        
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        
        # 2. Controller Architecture
        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_ctrl, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.pre_ln = nn.LayerNorm(d_ctrl)
        self.post_ln = nn.LayerNorm(d_ctrl) 

        # 3. Output Heads
        self.entropy_gate_module = VectorizedEntropyGate(d_ctrl, L)
        
        self.halting_head_module = VectorizedHaltingHead(
            input_dim=d_ctrl + 1, 
            hidden_dim=halting_hidden_dim, 
            L=L
        )
        
        self.classifier_heads = nn.ModuleList([
            nn.Linear(d_ctrl, self.num_classes) for _ in range(L)
        ])

        # 4. Initialization
        self.apply(self._init_weights)

        # 5. Override Halting Bias
        with torch.no_grad():
            self.halting_head_module.net2.bias.fill_(halting_bias_init)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, teacher_cls: torch.Tensor):
        B, L, D = teacher_cls.shape
        assert L == self.L

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
        all_gated_entropies = self.entropy_gate_module(z, all_entropies)

        combined_input = torch.cat([z, all_gated_entropies], dim=-1)
        halting_logits = self.halting_head_module(combined_input)
        
        class_logits = torch.stack(class_logits_list, dim=1)
        
        return halting_logits, class_logits, z