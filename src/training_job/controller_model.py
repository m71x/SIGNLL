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
            norm_first=True,  # <-- MODIFIED: Set to True for pre-normalization (more stable)
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
            # Use Xavier/Glorot uniform initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # Initialize biases to zero, except for the halting head (handled above)
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            # Use a standard normal distribution for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm weights to 1 and biases to 0
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
        # (The rest of the file remains unchanged)
# ...