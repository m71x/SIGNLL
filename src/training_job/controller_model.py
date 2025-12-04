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
    # x shape: [B, L, D]
    # cos/sin shape: [1, L, D]
    
    # FIX: Removed manual unsqueeze/expand. 
    # PyTorch broadcasting automatically handles [B, L, D] * [1, L, D]
    return (x * cos) + (rotate_half(x) * sin)

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
        # self.layer_embed removed (replaced by RoPE)
        
        # --- RoPE Initialization (FIXED) ---
        # We generate frequencies for pairs of values, so we need d_ctrl/2 frequencies.
        # arange(0, d_ctrl, 2) produces exactly d_ctrl/2 steps.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_ctrl, 2).float() / d_ctrl))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(L, dtype=torch.float32)
        # Outer product: [L, d_ctrl/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Concatenate to match full dimension [L, d_ctrl]
        emb = torch.cat((freqs, freqs), dim=-1) 
        
        # Reshape to [1, L, d_ctrl] for broadcasting against batch
        cos_cached = emb.cos().unsqueeze(0) 
        sin_cached = emb.sin().unsqueeze(0) 
        
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        # -----------------------------------
        
        # 2. Controller Architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_ctrl,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pre_ln = nn.LayerNorm(d_ctrl)
        self.post_ln = nn.LayerNorm(d_ctrl) 

        # 3. Output Heads
        self.halting_heads = nn.ModuleList([
            nn.Linear(d_ctrl, 1) for _ in range(L)
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
        """
        teacher_cls: [B, L, D_teacher]
        """
        B, L, D = teacher_cls.shape

        assert L == self.L and D == self.d_teacher, \
            f"Shape mismatch! got (L={L}, D={D}) but model expects (L={self.L}, D={self.d_teacher})"
        
        # 1. Controller Body Computation
        x = self.input_ln(teacher_cls)
        x = self.proj(x)
        
        # Apply RoPE (Broadcasting automatically handles [B, L, D] * [1, L, D])
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        
        x = self.pre_ln(x)
        
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        z = self.transformer(x, mask=attn_mask)
        z = self.post_ln(z) 
        
        # 2. Apply Unique Heads
        halting_logits_list = []
        class_logits_list = []
        
        for l in range(L):
            h_l = self.halting_heads[l](z[:, l, :]) 
            halting_logits_list.append(h_l)
            
            c_l = self.classifier_heads[l](z[:, l, :]) 
            class_logits_list.append(c_l)
        
        halting_logits = torch.cat(halting_logits_list, dim=-1).squeeze(-1) 
        class_logits = torch.stack(class_logits_list, dim=1)
        
        return halting_logits, class_logits, z

def compute_q_from_h(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    h = h.clamp(min=eps, max=1.0 - eps)
    one_minus_h = 1.0 - h
    S = torch.cumprod(one_minus_h, dim=1)
    ones = torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype)
    S_prev = torch.cat([ones, S[:, :-1]], dim=1)
    return S_prev * h