"""
Strategy II: The Representation Wormhole — Closed-Form Alignment
================================================================
Computes affine projection matrices W_L, b_L for each target layer L
such that: h_44_approx = h_L @ W_L + b_L

Uses Ridge Regression (OLS with L2 regularization) on the existing
phase2a_hidden_states.npz data. No TPU/GPU needed — runs on CPU.

Output: wormhole_projections.npz containing W_L, b_L for each layer.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy import linalg
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_LAYERS, PHASE2A_HIDDEN_PATH

# Where to find/save files — relative to script directory or CWD
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for candidate in [os.getcwd(), PROJECT_ROOT, os.path.expanduser("~/SIGNLL")]:
    if os.path.exists(os.path.join(candidate, PHASE2A_HIDDEN_PATH)):
        PROJECT_ROOT = candidate
        break

HIDDEN_PATH = os.path.join(PROJECT_ROOT, PHASE2A_HIDDEN_PATH)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "wormhole_projections.npz")

print("=" * 70)
print("  REPRESENTATION WORMHOLE: Closed-Form Layer Alignment")
print("=" * 70)

# Load hidden states
print(f"\nLoading {HIDDEN_PATH}...")
data = np.load(HIDDEN_PATH, allow_pickle=False)
hidden_states = data["hidden_states"]  # (N, 5120)
layer_indices = data["layer_indices"]   # (N,)
positions = data["positions"]           # (N,)

N = len(layer_indices)
hidden_dim = hidden_states.shape[1]
n_layers = len(TARGET_LAYERS)
n_groups = N // n_layers

print(f"  Samples: {N}, Hidden dim: {hidden_dim}")
print(f"  Groups: {n_groups}, Layers: {TARGET_LAYERS}")

# Reshape into (n_groups, n_layers, hidden_dim)
hs_grouped = hidden_states.reshape(n_groups, n_layers, hidden_dim)

# Final layer hidden states (layer 44, index -1)
h_final = hs_grouped[:, -1, :]  # (n_groups, hidden_dim)
print(f"  Final layer (44) hidden states: {h_final.shape}")

# Ridge regression parameter
LAMBDA = 1e-3

# Train/val split
n_train = int(n_groups * 0.8)
n_val = n_groups - n_train
perm = np.random.RandomState(42).permutation(n_groups)
train_idx = perm[:n_train]
val_idx = perm[n_train:]

print(f"  Train: {n_train}, Val: {n_val}")
print(f"  Ridge lambda: {LAMBDA}")

save_dict = {}

print(f"\n  {'Layer':>6} {'Time':>6} {'Train R²':>9} {'Val R²':>8} "
      f"{'Val MSE':>10} {'Val CosSim':>10}")
print(f"  {'─'*52}")

for l_idx, layer in enumerate(TARGET_LAYERS[:-1]):  # skip layer 44 (final)
    t0 = time.time()

    X = hs_grouped[:, l_idx, :]  # (n_groups, hidden_dim)
    Y = h_final                   # (n_groups, hidden_dim)

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # Add bias column: X_aug = [X, 1]
    X_train_aug = np.hstack([X_train, np.ones((n_train, 1))])
    X_val_aug = np.hstack([X_val, np.ones((n_val, 1))])

    # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
    # Using scipy.linalg.solve for numerical stability
    A = X_train_aug.T @ X_train_aug + LAMBDA * np.eye(hidden_dim + 1)
    B = X_train_aug.T @ Y_train
    W_aug = linalg.solve(A, B, assume_a='pos')  # (hidden_dim+1, hidden_dim)

    W = W_aug[:-1, :]  # (hidden_dim, hidden_dim)
    b = W_aug[-1, :]   # (hidden_dim,)

    # Evaluate
    Y_train_pred = X_train @ W + b
    Y_val_pred = X_val @ W + b

    # R² score
    ss_res_train = np.sum((Y_train - Y_train_pred) ** 2)
    ss_tot_train = np.sum((Y_train - np.mean(Y_train, axis=0)) ** 2)
    r2_train = 1 - ss_res_train / ss_tot_train

    ss_res_val = np.sum((Y_val - Y_val_pred) ** 2)
    ss_tot_val = np.sum((Y_val - np.mean(Y_val, axis=0)) ** 2)
    r2_val = 1 - ss_res_val / ss_tot_val

    # MSE
    val_mse = np.mean((Y_val - Y_val_pred) ** 2)

    # Cosine similarity
    cos_sims = np.sum(Y_val * Y_val_pred, axis=1) / (
        np.linalg.norm(Y_val, axis=1) * np.linalg.norm(Y_val_pred, axis=1) + 1e-8
    )
    val_cos = np.mean(cos_sims)

    elapsed = time.time() - t0

    print(f"  {layer:>6} {elapsed:>5.1f}s {r2_train:>9.4f} {r2_val:>8.4f} "
          f"{val_mse:>10.6f} {val_cos:>10.4f}")

    save_dict[f"W_{layer}"] = W.astype(np.float32)
    save_dict[f"b_{layer}"] = b.astype(np.float32)

# Save projections
np.savez(OUTPUT_PATH, **save_dict)
print(f"\n  Saved wormhole projections to {OUTPUT_PATH}")
print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")

# Now measure argmax agreement improvement
# Load the model's lm_head weights to compute logits
print(f"\n{'='*70}")
print("  ARGMAX AGREEMENT: Before vs After Wormhole Projection")
print(f"{'='*70}")

# We need the lm_head to compute logits. Check if we have it cached.
LM_HEAD_PATH = os.path.join(PROJECT_ROOT, "lm_head_weights.npz")
if os.path.exists(LM_HEAD_PATH):
    lm_head_data = np.load(LM_HEAD_PATH)
    lm_head_w = lm_head_data["weight"]  # (vocab_size, hidden_dim)
    print(f"  Loaded lm_head from {LM_HEAD_PATH}: {lm_head_w.shape}")

    # Compute agreement for each layer
    print(f"\n  {'Layer':>6} {'Baseline Agree%':>16} {'Wormhole Agree%':>16} {'Improvement':>12}")
    print(f"  {'─'*52}")

    # Final layer logits → argmax tokens
    final_logits = h_final[val_idx] @ lm_head_w.T  # (n_val, vocab_size)
    final_tokens = np.argmax(final_logits, axis=1)

    for l_idx, layer in enumerate(TARGET_LAYERS[:-1]):
        X_val = hs_grouped[val_idx, l_idx, :]

        # Baseline: raw hidden state → lm_head
        baseline_logits = X_val @ lm_head_w.T
        baseline_tokens = np.argmax(baseline_logits, axis=1)
        baseline_agree = np.mean(baseline_tokens == final_tokens) * 100

        # Wormhole: projected hidden state → lm_head
        W = save_dict[f"W_{layer}"]
        b = save_dict[f"b_{layer}"]
        projected = X_val @ W + b
        wormhole_logits = projected @ lm_head_w.T
        wormhole_tokens = np.argmax(wormhole_logits, axis=1)
        wormhole_agree = np.mean(wormhole_tokens == final_tokens) * 100

        improvement = wormhole_agree - baseline_agree
        print(f"  {layer:>6} {baseline_agree:>15.1f}% {wormhole_agree:>15.1f}% {improvement:>+11.1f}%")
else:
    print(f"\n  lm_head weights not found at {LM_HEAD_PATH}")
    print(f"  To measure argmax agreement, extract lm_head weights:")
    print(f"  Run extract_lm_head.py to save the weights, then re-run this script.")

print(f"\n{'='*70}")
print("  WORMHOLE COMPUTATION COMPLETE")
print(f"{'='*70}")
