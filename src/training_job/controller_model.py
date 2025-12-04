import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import math # Added for RoPE frequency calculation

# =========================================================================
# RoPE Utility Functions (Adapted for [B, L, D] input)
# =========================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimension for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to the input tensor x. Shapes: [B, L, D]"""
    # x shape: [batch, sequence, dim]
    # cos/sin shape: [1, sequence, dim]
    
    # RoPE is usually applied to Q/K, but here we apply it to the input 
    # as a replacement for absolute positional embedding.
    
    # Expand cos/sin to batch dimension if necessary
    cos = cos.unsqueeze(0).expand(x.size(0), -1, -1)
    sin = sin.unsqueeze(0).expand(x.size(0), -1, -1)

    return (x * cos) + (rotate_half(x) * sin)

# =========================================================================

class Controller(nn.Module):
    def __init__(
        self,
        L: int = 24,
        d_teacher: int = 1024,
        d_ctrl: int = 256, #consider changing to 512
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
        # self.layer_embed removed here
        
        # RoPE Initialization (NEW)
        # Rotary dimensions usually half the total dimension
        dim = d_ctrl // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(L, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # [L, D_ctrl]
        
        # Reshape for broadcasting with input [B, L, D_ctrl]
        cos_cached = emb.cos().unsqueeze(0) # [1, L, D_ctrl]
        sin_cached = emb.sin().unsqueeze(0) # [1, L, D_ctrl]
        
        # Cache cos/sin in buffers
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        
        # 2. Controller Architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_ctrl,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # MODIFIED: Use pre-normalization for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pre_ln = nn.LayerNorm(d_ctrl)
        self.post_ln = nn.LayerNorm(d_ctrl) 

        # 3. Output Heads: 24 UNIQUE SETS OF HEADS
        
        # Halting Heads: A list of L unique nn.Linear(d_ctrl, 1) layers
        self.halting_heads = nn.ModuleList([
            nn.Linear(d_ctrl, 1) for _ in range(L)
        ])
        
        # Classification Heads: A list of L unique nn.Linear(d_ctrl, num_classes) layers
        self.classifier_heads = nn.ModuleList([
            nn.Linear(d_ctrl, self.num_classes) for _ in range(L)
        ])

        # 4. Apply Custom Initialization
        # This will initialize all Linear, Embedding, and LayerNorm modules
        self.apply(self._init_weights)

        # 5. Override Halting Bias Initialization
        # This MUST come after self.apply() to override the bias=0 from _init_weights
        with torch.no_grad():
            for head in self.halting_heads:
                head.bias.fill_(halting_bias_init)

    def _init_weights(self, module):
        """Applies stable weight initialization."""
        if isinstance(module, nn.Linear):
            # CHANGE: Use Kaiming init instead of Xavier for GELU/ReLU stability
            # 'fan_in' preserves magnitude in the forward pass.
            # 'nonlinearity="relu"' is the standard approximation for GELU.
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            # Keep small normal init for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, teacher_cls: torch.Tensor):
        """
        teacher_cls: [B, L, D_teacher]
        Returns:
            halting_logits: [B, L]
            class_logits: [B, L, num_classes]
            z: [B, L, d_ctrl]
        """
        B, L, D = teacher_cls.shape

        assert L == self.L and D == self.d_teacher, \
            f"Shape mismatch! got (L={L}, D={D}) but model expects (L={self.L}, D={self.d_teacher})"
        
        
        # 1. Controller Body Computation
        x = self.input_ln(teacher_cls) # Input LayerNorm (from previous change)
        x = self.proj(x)
        
        # Apply RoPE (Replaces Absolute Positional Embedding Addition)
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        
        x = self.pre_ln(x)
        
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        z = self.transformer(x, mask=attn_mask)
        z = self.post_ln(z) # z is the sequence of layer representations [B, L, d_ctrl]
        
        # 2. Apply Unique Heads (Iterate over the L tokens)
        halting_logits_list = []
        class_logits_list = []
        
        # z[:, l, :] is the l-th depth token y_l (shape [B, d_ctrl])
        for l in range(L):
            # Apply the l-th unique halting head
            h_l = self.halting_heads[l](z[:, l, :]) # Output shape: [B, 1]
            halting_logits_list.append(h_l)
            
            # Apply the l-th unique classification head
            c_l = self.classifier_heads[l](z[:, l, :]) # Output shape: [B, num_classes]
            class_logits_list.append(c_l)
        
        # Stack the results to match the required output shape
        # Halting logits: L lists of [B, 1] -> [B, L, 1] -> [B, L]
        halting_logits = torch.cat(halting_logits_list, dim=-1).squeeze(-1) 
        
        # Class logits: L lists of [B, num_classes] -> [B, L, num_classes]
        class_logits = torch.stack(class_logits_list, dim=1)
        
        return halting_logits, class_logits, z

# The utility function remains unchanged as it operates on the outputs, not the model.
def compute_q_from_h(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    h = h.clamp(min=eps, max=1.0 - eps)
    one_minus_h = 1.0 - h
    S = torch.cumprod(one_minus_h, dim=1)
    ones = torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype)
    S_prev = torch.cat([ones, S[:, :-1]], dim=1)
    return S_prev * h