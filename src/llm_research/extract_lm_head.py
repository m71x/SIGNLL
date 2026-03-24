"""
Extract lm_head weights from the model for offline evaluation.
Runs on a single TPU worker, saves to lm_head_weights.npz.
"""
import sys
import os
sys.stdout.reconfigure(line_buffering=True)

# Force single-host TPU
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_ID

import numpy as np
from transformers import AutoModelForCausalLM
import torch

print("Loading model to extract lm_head weights...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

# Extract lm_head weight matrix
lm_head_w = model.lm_head.weight.detach().cpu().numpy()  # (vocab_size, hidden_dim)
print(f"  lm_head weight shape: {lm_head_w.shape}")
print(f"  dtype: {lm_head_w.dtype}")

# Also extract the final layer norm (needed to properly compute logits)
if hasattr(model.model, 'norm'):
    norm_weight = model.model.norm.weight.detach().cpu().numpy()
    norm_bias = None
    if hasattr(model.model.norm, 'bias') and model.model.norm.bias is not None:
        norm_bias = model.model.norm.bias.detach().cpu().numpy()
    print(f"  Final LayerNorm weight shape: {norm_weight.shape}")

    save_dict = {"weight": lm_head_w, "norm_weight": norm_weight}
    if norm_bias is not None:
        save_dict["norm_bias"] = norm_bias
else:
    save_dict = {"weight": lm_head_w}

output_path = "lm_head_weights.npz"
np.savez(output_path, **save_dict)
print(f"  Saved to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

del model
print("Done.")
