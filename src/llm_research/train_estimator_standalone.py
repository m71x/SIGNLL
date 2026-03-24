"""
Standalone Phase 3: Train regret estimator WITHOUT distributed JAX.
Runs on a single worker — no TPU coordination needed.
"""
import sys
import os
sys.stdout.reconfigure(line_buffering=True)

# Force CPU-only JAX — no TPU coordination needed for estimator training
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import numpy as np
import gc
import os

from config import (
    TARGET_LAYERS, ESTIMATOR_LR, ESTIMATOR_EPOCHS, ESTIMATOR_BATCH,
    ESTIMATOR_PATIENCE, ESTIMATOR_DROPOUT, ESTIMATOR_WEIGHT_DECAY,
    LABEL_SMOOTHING, TRAIN_SPLIT, REGRET_DATA_PATH, ESTIMATOR_PATH,
    V3_PROGRESSIVE, V3_KEEP_FEATURES, V3_NUM_KEPT_FEATURES,
    V3_PROJ_DIM, V3_CONV_DIM, V3_CONV_KERNEL,
    ASYM_OVER_PENALTY, ASYM_UNDER_PENALTY,
)

import optax
from regret_estimator import (create_estimator, create_progressive_estimator,
                              save_estimator)
from flax import nnx

print(f"{'='*70}")
print("PHASE 3: Training regret estimator (standalone, no distributed JAX)")
print(f"{'='*70}")
print(f"  JAX devices: {jax.devices()}")

# Load regret dataset
data = np.load(REGRET_DATA_PATH, allow_pickle=False)
hs = data["hidden_states"]
if hs.dtype.kind not in ('f', 'i', 'u'):
    hs_uint16 = hs.view(np.uint16)
    hs = np.array(jnp.array(hs_uint16).view(jnp.bfloat16).astype(jnp.float32))
hidden_states = jnp.array(hs, dtype=jnp.float32)
layer_indices = jnp.array(data["layer_indices"])
positions = jnp.array(data["positions"])
regret_targets = jnp.array(data["regret_values"])

# Load logit features
if "entropies" in data:
    entropies_raw = jnp.array(data["entropies"], dtype=jnp.float32)
    margins_raw = jnp.array(data["margins"], dtype=jnp.float32)
    has_logit_features = True
else:
    has_logit_features = False
    entropies_raw = jnp.zeros(len(regret_targets))
    margins_raw = jnp.zeros(len(regret_targets))

# V2 features
has_v2_features = "v2_features" in data
if has_v2_features:
    v2_extra_raw = jnp.array(data["v2_features"], dtype=jnp.float32)
    num_meta = 4 + v2_extra_raw.shape[1]
    print(f"  V2 features detected: {v2_extra_raw.shape[1]} extra features (total meta={num_meta})")
else:
    v2_extra_raw = None
    num_meta = 4

N = hidden_states.shape[0]
hidden_dim = hidden_states.shape[1]
print(f"  Dataset: {N} samples, hidden_dim={hidden_dim}")

if has_v2_features:
    binary_raw = (regret_targets > 0.5).astype(jnp.float32)
    training_targets = regret_targets
    nonzero_frac = float(jnp.mean(binary_raw))
    print(f"  V2 KL targets: mean={float(jnp.mean(regret_targets)):.4f}, "
          f"std={float(jnp.std(regret_targets)):.4f}")
    print(f"  Not converged (>0.5): {nonzero_frac:.3f} ({int(jnp.sum(binary_raw))}/{N})")
else:
    binary_raw = (regret_targets > 0).astype(jnp.float32)
    training_targets = binary_raw * (1 - LABEL_SMOOTHING) + (1 - binary_raw) * LABEL_SMOOTHING
    nonzero_frac = float(jnp.mean(binary_raw))

# Per-layer normalization
norm_stats = {}
unique_layers = sorted(set(int(x) for x in np.unique(np.array(layer_indices))))
print(f"  Computing per-layer normalization stats...")
for layer in unique_layers:
    mask = np.array(layer_indices) == layer
    layer_hs = np.array(hidden_states[mask])
    mean = np.mean(layer_hs, axis=0)
    std = np.std(layer_hs, axis=0)
    norm_stats[f"layer_{layer}_mean"] = mean
    norm_stats[f"layer_{layer}_std"] = std
    print(f"    Layer {layer}: mean_norm={np.linalg.norm(mean):.2f}, "
          f"mean_std={np.mean(std):.4f}, min_std={np.min(std):.6f}")
    hidden_states = hidden_states.at[mask].set(
        (hidden_states[mask] - jnp.array(mean)) / (jnp.array(std) + 1e-8)
    )

# Normalize entropy and margin
ent_mean = float(jnp.mean(entropies_raw))
ent_std = float(jnp.std(entropies_raw))
mar_mean = float(jnp.mean(margins_raw))
mar_std = float(jnp.std(margins_raw))
norm_stats["entropy_mean"] = ent_mean
norm_stats["entropy_std"] = ent_std
norm_stats["margin_mean"] = mar_mean
norm_stats["margin_std"] = mar_std
entropies_norm = ((entropies_raw - ent_mean) / (ent_std + 1e-8)).reshape(-1, 1)
margins_norm = ((margins_raw - mar_mean) / (mar_std + 1e-8)).reshape(-1, 1)

# V2 feature normalization
v2_extra_norm = None
v2_feature_names = ["agreement", "confidence", "kl_div", "int_entropy",
                    "cos_sim", "rel_change", "proj_final",
                    "is_syntax", "is_whitespace"]
if has_v2_features:
    v2_extra_norm = v2_extra_raw.copy()
    for i, fname in enumerate(v2_feature_names):
        col = v2_extra_raw[:, i]
        col_mean = float(jnp.mean(col))
        col_std = float(jnp.std(col))
        norm_stats[f"v2_{fname}_mean"] = col_mean
        norm_stats[f"v2_{fname}_std"] = col_std
        if col_std > 1e-6:
            v2_extra_norm = v2_extra_norm.at[:, i].set((col - col_mean) / (col_std + 1e-8))
        print(f"    V2 {fname}: mean={col_mean:.4f}, std={col_std:.4f}")

# Normalize layer/position
num_layers = max(int(jnp.max(layer_indices)) + 1, 1)
max_pos = max(int(jnp.max(positions)) + 1, 1)
layer_norm_feat = layer_indices.astype(jnp.float32).reshape(-1, 1) / (num_layers - 1)
pos_norm = positions.astype(jnp.float32).reshape(-1, 1) / (max_pos - 1)

# Progressive architecture
use_progressive = V3_PROGRESSIVE and has_v2_features
if use_progressive:
    num_target_layers = len(TARGET_LAYERS)
    print(f"\n  V3 PROGRESSIVE MODE: Conv1D over {num_target_layers} layers")

    assert N % num_target_layers == 0
    num_groups = N // num_target_layers

    layer_arr = np.array(layer_indices)
    layer_groups = layer_arr.reshape(num_groups, num_target_layers)
    for g in range(min(10, num_groups)):
        assert np.all(np.diff(layer_groups[g]) > 0)
    print(f"  Verified {num_groups} position groups x {num_target_layers} layers")

    v2_kept = v2_extra_norm[:, V3_KEEP_FEATURES]
    print(f"  Keeping {V3_NUM_KEPT_FEATURES} V2 features: "
          f"{[v2_feature_names[i] for i in V3_KEEP_FEATURES]}")

    meta_per_layer = 4 + V3_NUM_KEPT_FEATURES
    feat_dim = hidden_dim + meta_per_layer
    print(f"  Feature dim per layer: {hidden_dim} (hidden) + {meta_per_layer} (meta) = {feat_dim}")

    all_feats_flat = jnp.concatenate([
        hidden_states, layer_norm_feat, pos_norm, entropies_norm, margins_norm, v2_kept,
    ], axis=-1)

    seq_features = all_feats_flat.reshape(num_groups, num_target_layers, feat_dim)
    seq_targets = training_targets.reshape(num_groups, num_target_layers)
    seq_binary = binary_raw.reshape(num_groups, num_target_layers)

    max_target_layer = max(TARGET_LAYERS)
    min_target_layer = min(TARGET_LAYERS)
    prog_layer_weights = jnp.array([
        2.0 - (tl - min_target_layer) / max(max_target_layer - min_target_layer, 1)
        for tl in TARGET_LAYERS
    ])
    print(f"  Layer weights: {TARGET_LAYERS[0]}→{float(prog_layer_weights[0]):.1f}, "
          f"{TARGET_LAYERS[-1]}→{float(prog_layer_weights[-1]):.1f}")

    # Stratified split
    group_max_target = jnp.max(seq_targets, axis=1)
    perm = np.random.RandomState(42).permutation(num_groups)
    fragile_idx = perm[np.array(group_max_target[perm]) > 0.5]
    safe_idx = perm[np.array(group_max_target[perm]) <= 0.5]
    f_split = int(len(fragile_idx) * TRAIN_SPLIT)
    s_split = int(len(safe_idx) * TRAIN_SPLIT)
    train_idx = np.concatenate([fragile_idx[:f_split], safe_idx[:s_split]])
    val_idx = np.concatenate([fragile_idx[f_split:], safe_idx[s_split:]])
    np.random.RandomState(42).shuffle(train_idx)
    np.random.RandomState(42).shuffle(val_idx)
    print(f"  Train: {len(train_idx)} groups (fragile: {f_split}), "
          f"Val: {len(val_idx)} groups (fragile: {len(fragile_idx)-f_split})")

    save_meta = {
        "arch": "progressive",
        "num_layers": num_target_layers,
        "num_meta_per_layer": meta_per_layer,
        "proj_dim": V3_PROJ_DIM,
        "conv_dim": V3_CONV_DIM,
        "kernel_size": V3_CONV_KERNEL,
    }

    # Convert to numpy
    seq_features = np.array(seq_features)
    seq_targets = np.array(seq_targets)
    seq_binary = np.array(seq_binary)
    prog_layer_weights_np = np.array(prog_layer_weights)

    model = create_progressive_estimator(
        hidden_dim, num_target_layers, seed=42,
        num_meta_per_layer=meta_per_layer,
        proj_dim=V3_PROJ_DIM, conv_dim=V3_CONV_DIM,
        kernel_size=V3_CONV_KERNEL, dropout_rate=ESTIMATOR_DROPOUT,
    )

    warmup_steps = 50
    batch_size = max(ESTIMATOR_BATCH // num_target_layers, 16)
    total_steps = ESTIMATOR_EPOCHS * (len(train_idx) // batch_size + 1)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, ESTIMATOR_LR, warmup_steps),
            optax.cosine_decay_schedule(ESTIMATOR_LR, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=ESTIMATOR_WEIGHT_DECAY),
                              wrt=nnx.Param)

    # Asymmetric MSE loss
    def loss_fn(model, features, targets, layer_weights):
        preds = model(features, deterministic=False).squeeze(-1)
        residuals = targets - preds
        asym_weights = jnp.where(residuals > 0, ASYM_OVER_PENALTY, ASYM_UNDER_PENALTY)
        mse = residuals ** 2 * asym_weights
        weighted = mse * layer_weights[None, :]
        return jnp.mean(weighted)

    def train_step(model, optimizer, features, targets, layer_weights):
        loss, grads = nnx.value_and_grad(loss_fn)(model, features, targets, layer_weights)
        optimizer.update(model, grads)
        return loss

    print(f"\n  Training PROGRESSIVE for up to {ESTIMATOR_EPOCHS} epochs (patience={ESTIMATOR_PATIENCE})")
    print(f"  Batch size: {batch_size} groups, LR: cosine from {ESTIMATOR_LR}")
    print(f"  Loss: Asymmetric MSE (over={ASYM_OVER_PENALTY}x, under={ASYM_UNDER_PENALTY}x), Dropout: {ESTIMATOR_DROPOUT}")
    print(f"  Training in eager mode (numpy data, auto-placed on device)")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(ESTIMATOR_EPOCHS):
        epoch_perm = np.random.RandomState(epoch).permutation(len(train_idx))
        shuffled = train_idx[epoch_perm]

        epoch_losses = []
        for batch_start in range(0, len(shuffled), batch_size):
            batch_end = min(batch_start + batch_size, len(shuffled))
            idx = shuffled[batch_start:batch_end]
            loss = train_step(
                model, optimizer,
                seq_features[idx], seq_targets[idx], prog_layer_weights_np,
            )
            epoch_losses.append(float(loss))

        train_loss = np.mean(epoch_losses)

        # Validation
        val_preds = model(seq_features[val_idx], deterministic=True).squeeze(-1)
        val_residuals = seq_targets[val_idx] - val_preds
        val_asym_w = jnp.where(val_residuals > 0, ASYM_OVER_PENALTY, ASYM_UNDER_PENALTY)
        val_mse = float(jnp.mean(
            val_residuals ** 2 * val_asym_w * prog_layer_weights_np[None, :]
        ))
        val_acc = float(jnp.mean((val_preds > 0.5) == seq_binary[val_idx]))

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_val_acc = val_acc
            patience_counter = 0
            save_estimator(model, ESTIMATOR_PATH, norm_stats=norm_stats, meta=save_meta)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter == 0:
            print(f"  Epoch {epoch+1:3d}: train_mse={train_loss:.6f}, val_mse={val_mse:.6f}, "
                  f"val_acc={val_acc:.3f}{' *best*' if patience_counter == 0 else ''}")

        if patience_counter >= ESTIMATOR_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {ESTIMATOR_PATIENCE} epochs)")
            break

    print(f"\n  Progressive training complete. Best val MSE: {best_val_loss:.6f}, "
          f"Best val acc: {best_val_acc:.3f}")
    print(f"  Model saved to {ESTIMATOR_PATH}.npz")
else:
    print("ERROR: Non-progressive mode not supported in standalone script")
    sys.exit(1)
