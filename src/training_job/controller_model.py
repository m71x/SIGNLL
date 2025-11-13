# controller_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(
        self,
        L: int = 24,
        d_teacher: int = 1024,
        d_ctrl: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        halting_bias_init: float = -2.5,
    ):
        super().__init__()
        self.L = L
        self.d_teacher = d_teacher
        self.d_ctrl = d_ctrl

        # projection from teacher CLS dim -> controller dim
        self.proj = nn.Linear(d_teacher, d_ctrl)

        # learned depth embeddings (instead of sinusoidal)
        self.layer_embed = nn.Embedding(L, d_ctrl)

        # Transformer encoder stack (Pre-LN style)
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

        # Heads: halting and classification
        self.halting_head = nn.Sequential(
            nn.Linear(d_ctrl, d_ctrl // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ctrl // 2, 1),
        )
        with torch.no_grad():
            self.halting_head[-1].bias.fill_(halting_bias_init)

        self.classifier_head = nn.Sequential(
            nn.Linear(d_ctrl, d_ctrl // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ctrl // 2, 1),
        )

        self.post_ln = nn.LayerNorm(d_ctrl)

    def forward(self, teacher_cls: torch.Tensor):
        """
        teacher_cls: [B, L, D_teacher]
        Returns:
            halting_logits: [B, L]
            class_logits: [B, L]
            z: [B, L, d_ctrl]
        """
        B, L, D = teacher_cls.shape
        assert L == self.L and D == self.d_teacher

        x = self.proj(teacher_cls)
        idx = torch.arange(self.L, device=teacher_cls.device).unsqueeze(0).expand(B, -1)
        x = x + self.layer_embed(idx)
        x = self.pre_ln(x)

        # causal mask (prevent attending to future layers)
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        z = self.transformer(x, mask=attn_mask)
        z = self.post_ln(z)

        halting_logits = self.halting_head(z).squeeze(-1)
        class_logits = self.classifier_head(z).squeeze(-1)
        return halting_logits, class_logits, z


def compute_q_from_h(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Given halting probabilities h [B, L], compute q_i = h_i * prod_{j < i} (1 - h_j)
    Returns q [B, L]
    """
    h = h.clamp(min=eps, max=1.0 - eps)
    one_minus_h = 1.0 - h
    S = torch.cumprod(one_minus_h, dim=1)
    ones = torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype)
    S_prev = torch.cat([ones, S[:, :-1]], dim=1)
    return S_prev * h
