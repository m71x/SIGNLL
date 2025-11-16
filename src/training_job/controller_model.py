import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

class Controller(nn.Module):
    def __init__(
        self,
        L: int = 24,
        d_teacher: int = 1024,
        d_ctrl: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.3,
        halting_bias_init: float = -2.5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.L = L
        self.d_teacher = d_teacher
        self.d_ctrl = d_ctrl
        self.num_classes = num_classes

        # 1. Inputs and Embeddings
        self.proj = nn.Linear(d_teacher, d_ctrl)
        self.layer_embed = nn.Embedding(L, d_ctrl)
        
        # 2. Controller Architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_ctrl,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
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
        
        # =========================================================================
        # --- FIX: Weight Re-scaling for Stability in Deep Networks (n_layers=12) ---
        # =========================================================================
        with torch.no_grad():
            
            # 1. Initialize Classification Heads
            # Scale down weights to prevent compounded deep-network output explosion
            # and set a small negative bias for stable initial probabilities.
            for head in self.classifier_heads:
                # Use a small gain (e.g., 1e-2) to keep initial logits near zero
                torch.nn.init.xavier_uniform_(head.weight, gain=1e-2) 
                if head.bias is not None:
                    # Set bias slightly negative for initial stability
                    torch.nn.init.constant_(head.bias, -0.1) 
                    
            # 2. Initialize Halting Heads
            for head in self.halting_heads:
                # Re-initialize weights to prevent large initial logits
                torch.nn.init.xavier_uniform_(head.weight, gain=1e-2) 
                # Keep the user's explicit bias initialization for the halting head
                head.bias.fill_(halting_bias_init) 
        # =========================================================================


    def forward(self, teacher_cls: torch.Tensor):
        """
        teacher_cls: [B, L, D_teacher]
        Returns:
            halting_logits: [B, L]
            class_logits: [B, L, num_classes]
            z: [B, L, d_ctrl]
        """
        #teacher_cls = teacher_cls[:, 1:25, :]   # now shape becomes [B, 24, D]
        B, L, D = teacher_cls.shape
        #xm.master_print(f"[DEBUG] forward() received shape: B={B}, L={L}, D={D}")
        #xm.master_print(f"[DEBUG] Model expects L={self.L}, D={self.d_teacher}")

        assert L == self.L and D == self.d_teacher, \
            f"Shape mismatch! got (L={L}, D={D}) but model expects (L={self.L}, D={self.d_teacher})"
        
        
        # 1. Controller Body Computation (Same as before)
        x = self.proj(teacher_cls)
        idx = torch.arange(self.L, device=teacher_cls.device).unsqueeze(0).expand(B, -1)
        x = x + self.layer_embed(idx)
        x = self.pre_ln(x)
        
        # Mask for the self-attention layer (causal mask: attention only to preceding layers)
        # Note: L must be equal to the sequence length in the batch (24)
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
    """
    Computes the probability of halting at layer l, q_l, given the probability of 
    halting *not before* layer l, h_l.
    h: [B, L], the per-layer halting probability (h_l = P(halt | l))
    Returns:
        q: [B, L], the probability of halting exactly at layer l (q_l = P(halt at l))
    """
    h = h.clamp(min=eps, max=1.0 - eps)
    # S is the probability of *not* halting before this layer: P(stop > l) = Prod_{i=1}^{l} (1 - h_i)
    one_minus_h = 1.0 - h
    S = torch.cumprod(one_minus_h, dim=1)
    
    # S_prev is P(stop > l-1)
    # [1, S_1, S_2, ..., S_{L-1}]
    ones = torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype)
    S_prev = torch.cat([ones, S[:, :-1]], dim=1)
    
    # q_l = P(halt at l) = P(halt | l) * P(stop > l-1) = h_l * S_prev_{l}
    return S_prev * h