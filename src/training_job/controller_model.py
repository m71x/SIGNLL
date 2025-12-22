import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import math 

# =========================================================================
# RoPE Utility Functions
# =========================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimension for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to the input tensor x. Shapes: [B, L, D]"""
    return (x * cos) + (rotate_half(x) * sin)

# =========================================================================
# SwiGLU Feed-Forward Network
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

# =========================================================================
# Custom Transformer Layer (Pre-Norm + SwiGLU)
# =========================================================================
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
# NEW: Simple Gating Module for Entropy
# =========================================================================
class VectorizedEntropyGate(nn.Module):
    def __init__(self, d_ctrl: int, L: int):
        super().__init__()
        # Use simple Linears; they natively support [Batch, Seq, Dim] inputs
        self.net = nn.Sequential(
            nn.Linear(d_ctrl, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        # z: [B, L, D]
        # entropy: [B, L, 1]
        
        gate_weight = self.net(z) # Output: [B, L, 1]
        return entropy * gate_weight
    
    def forward(self, z: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, d_ctrl] - hidden state
            entropy: [B, 1] - raw entropy value
        Returns:
            gated_entropy: [B, 1] - entropy scaled by learned gate
        """
        gate_weight = self.gate(z)  # [B, 1]
        return entropy * gate_weight

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
    ):
        super().__init__()
        self.L = L
        self.d_teacher = d_teacher
        self.d_ctrl = d_ctrl
        self.num_classes = num_classes

        # 1. Inputs and Embeddings
        self.input_ln = nn.LayerNorm(d_teacher)
        self.proj = nn.Linear(d_teacher, d_ctrl)
        
        # RoPE Initialization
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
            CustomTransformerLayer(
                d_model=d_ctrl,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.pre_ln = nn.LayerNorm(d_ctrl)
        self.post_ln = nn.LayerNorm(d_ctrl) 

        # 3. Output Heads
        # NEW: Add entropy gates (one per layer)
        self.entropy_gates = nn.ModuleList([
            EntropyGate(d_ctrl) for _ in range(L)
        ])
        
        # Halting heads now take [d_ctrl + 1] (hidden + gated_entropy)
        self.halting_heads = nn.ModuleList([
            nn.Linear(d_ctrl + 1, 1) for _ in range(L)
        ])
        
        self.classifier_heads = nn.ModuleList([
            nn.Linear(d_ctrl, self.num_classes) for _ in range(L)
        ])

        # 4. Initialization
        self.apply(self._init_weights)

        # 5. Override Halting Bias
        with torch.no_grad():
            for head in self.halting_heads:
                head.bias.fill_(halting_bias_init)
            
            # Initialize gates to output ~0.5 initially (middle of sigmoid range)
            for gate_module in self.entropy_gates:
                gate_module.gate[-2].bias.fill_(0.0)  # Sigmoid(0) = 0.5

    def _init_weights(self, module):
        """Applies stable weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, teacher_cls: torch.Tensor):
        B, L, D = teacher_cls.shape

        assert L == self.L and D == self.d_teacher, \
            f"Shape mismatch! got (L={L}, D={D}) but model expects (L={self.L}, D={self.d_teacher})"
        
        # 1. Controller Body Computation
        x = self.input_ln(teacher_cls)
        x = self.proj(x)
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        x = self.pre_ln(x)
        
        # Mask
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        
        # Iterate through Custom Layers
        z = x
        for layer in self.layers:
            z = layer(z, mask=attn_mask)
            
        z = self.post_ln(z) 
        
        # 2. Apply Unique Heads
        halting_logits_list = []
        class_logits_list = []
        
        for l in range(L):
            # A. Compute Classifier Logits
            c_l = self.classifier_heads[l](z[:, l, :]) 
            class_logits_list.append(c_l)
            
            # B. Calculate Entropy (numerically stable)
            log_probs = F.log_softmax(c_l, dim=-1)
            probs = torch.exp(log_probs)
            
            # Clamp to avoid log(0)
            probs_safe = probs.clamp(min=1e-10, max=1.0)
            log_probs_safe = torch.log(probs_safe)
            
            # Entropy: -sum(p * log(p))
            entropy = -(probs_safe * log_probs_safe).sum(dim=-1, keepdim=True)  # [B, 1]
            
            # Normalize entropy to [0, 1] range
            max_entropy = math.log(self.num_classes)
            entropy_normalized = (entropy / max_entropy).clamp(0.0, 1.0)
            
            # C. NEW: Apply learned gate to entropy
            gated_entropy = self.entropy_gates[l](z[:, l, :], entropy_normalized)
            
            # D. Concatenate [Hidden_State, Gated_Entropy] for Halting Head
            combined_input = torch.cat([z[:, l, :], gated_entropy], dim=-1)
            
            # E. Compute Halting Logits
            h_l = self.halting_heads[l](combined_input) 
            halting_logits_list.append(h_l)
        
        halting_logits = torch.cat(halting_logits_list, dim=-1)  # [B, L]
        class_logits = torch.stack(class_logits_list, dim=1)  # [B, L, num_classes]
        
        return halting_logits, class_logits, z

def compute_q_from_h(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    h = h.clamp(min=eps, max=1.0 - eps)
    one_minus_h = 1.0 - h
    S = torch.cumprod(one_minus_h, dim=1)
    ones = torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype)
    S_prev = torch.cat([ones, S[:, :-1]], dim=1)
    return S_prev * h