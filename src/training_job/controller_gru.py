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
# OPTIMIZED: Vectorized Gating Module
# =========================================================================
class VectorizedEntropyGate(nn.Module):
    def __init__(self, d_ctrl: int, L: int):
        super().__init__()
        self.L = L
        self.d_ctrl = d_ctrl
        
        # Layer 1: d_ctrl -> 8 (Unique weights per layer L)
        self.net1 = nn.Conv1d(
            in_channels=d_ctrl * L, 
            out_channels=8 * L, 
            kernel_size=1, 
            groups=L
        )
        self.act = nn.Tanh()
        
        # Layer 2: 8 -> 1 (Unique weights per layer L)
        self.net2 = nn.Conv1d(
            in_channels=8 * L, 
            out_channels=1 * L, 
            kernel_size=1, 
            groups=L
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.net2.bias, 0.0)

    def forward(self, z, entropy):
        B, L, D = z.shape
        z_reshaped = z.transpose(1, 2).reshape(B, D * L, 1)
        out = self.net1(z_reshaped)
        out = self.act(out)
        out = self.net2(out)
        gate = self.sigmoid(out).reshape(B, L, 1)
        return entropy * gate

# =========================================================================
# CONTROLLER (Recurrent Architecture)
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
        d_halt_hidden: int = 64  # NEW: Hidden dimension for GRU
    ):
        super().__init__()
        self.L = L
        self.d_teacher = d_teacher
        self.d_ctrl = d_ctrl
        self.d_halt_hidden = d_halt_hidden
        self.num_classes = num_classes

        # 1. Inputs and Embeddings
        self.input_ln = nn.LayerNorm(d_teacher)
        self.proj = nn.Linear(d_teacher, d_ctrl)
        
        # RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_ctrl, 2).float() / d_ctrl))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(L, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) 
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)
        
        # 2. Transformer Backbone
        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_ctrl, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pre_ln = nn.LayerNorm(d_ctrl)
        self.post_ln = nn.LayerNorm(d_ctrl) 

        # 3. Classifiers & Gating
        self.entropy_gate_module = VectorizedEntropyGate(d_ctrl, L)
        self.classifier_heads = nn.ModuleList([
            nn.Linear(d_ctrl, self.num_classes) for _ in range(L)
        ])
        
        # 4. NEW: Recurrent Halting Logic
        # GRU Cell: Input = [Backbone(512) + GatedEntropy(1)]
        self.halting_gru = nn.GRUCell(d_ctrl + 1, d_halt_hidden)
        
        # Stabilization: LayerNorm on Hidden State
        self.halt_ln = nn.LayerNorm(d_halt_hidden)
        
        # Residual Halting Head: Input = [HiddenState(64) + Delta(64)]
        self.halting_proj = nn.Linear(d_halt_hidden * 2, 1)

        # Initialization
        self.apply(self._init_weights)

        # Specialized Recurrent Init (Orthogonal for stability)
        nn.init.orthogonal_(self.halting_gru.weight_hh)
        nn.init.orthogonal_(self.halting_gru.weight_ih)

        # Override Halting Bias
        with torch.no_grad():
            self.halting_proj.bias.fill_(halting_bias_init)

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

        # --- 1. Backbone Forward ---
        x = self.input_ln(teacher_cls)
        x = self.proj(x)
        x = apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
        x = self.pre_ln(x)
        attn_mask = torch.triu(torch.ones(L, L, device=teacher_cls.device), diagonal=1).bool()
        
        z = x
        for layer in self.layers:
            z = layer(z, mask=attn_mask)
        z = self.post_ln(z) 
        
        # --- 2. Class Logits & Entropy ---
        class_logits_list = []
        entropy_list = []
        max_entropy = math.log(self.num_classes)
        
        for l in range(L):
            c_l = self.classifier_heads[l](z[:, l, :]) 
            class_logits_list.append(c_l)
            
            # Stable Entropy
            log_probs = F.log_softmax(c_l, dim=-1)
            probs = torch.exp(log_probs).clamp(min=1e-10, max=1.0)
            ent = -(probs * log_probs).sum(dim=-1, keepdim=True)
            entropy_list.append((ent / max_entropy).clamp(0.0, 1.0))
            
        all_entropies = torch.stack(entropy_list, dim=1)
        
        # Vectorized Gating
        all_gated_entropies = self.entropy_gate_module(z, all_entropies)

        # --- 3. Recurrent Halting (The Manager) ---
        # Input construction: [B, L, D+1]
        gru_inputs = torch.cat([z, all_gated_entropies], dim=-1)
        
        halting_logits_list = []
        saved_states_list = []
        
        # Initial State: [B, d_hidden]
        h_prev = torch.zeros(B, self.d_halt_hidden, device=z.device)
        
        for l in range(L):
            input_l = gru_inputs[:, l, :] # [B, D+1]
            
            # a. Update State
            h_curr = self.halting_gru(input_l, h_prev)
            
            # b. Stabilize State (Prevent drift over 24 layers)
            h_curr = self.halt_ln(h_curr)
            
            # c. Calculate Innovation (Delta)
            delta = h_curr - h_prev
            
            # d. Halting Decision: Based on State + Delta
            # [B, d_hidden * 2]
            head_input = torch.cat([h_curr, delta], dim=-1)
            h_logit = self.halting_proj(head_input)
            
            halting_logits_list.append(h_logit)
            saved_states_list.append(h_curr)
            
            # Update for next step
            h_prev = h_curr
        
        halting_logits = torch.cat(halting_logits_list, dim=-1) 
        class_logits = torch.stack(class_logits_list, dim=1)
        
        # RETURN SAVED STATES for Diversity Loss in Stage 2
        return halting_logits, class_logits, z, saved_states_list