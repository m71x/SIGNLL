"""
Regret Estimator — Flax MLP Model
===================================
Lightweight MLP that predicts whether a token position is "fragile"
(perturbation causes regret) from hidden states, layer index, and position.

Architecture (v2 — binary classification with per-layer normalization):
    Input: normalized hidden_state (hidden_dim) + layer_idx (1) + position (1)
    → Linear(hidden_dim+2, 512) → GELU → Dropout(0.1)
    → Linear(512, 128) → GELU → Dropout(0.1)
    → Linear(128, 1) → Sigmoid
    Output: P(fragile) ∈ [0, 1]

Key changes from v1:
    - Per-layer hidden state normalization (fixes layer 40 dead neurons)
    - Binary classification with BCE loss (matches data: 82% regret=1.0)
    - Dropout for regularization (train/val gap was large)
    - Wider bottleneck (512→128 vs 256→64)
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


class RegretEstimator(nnx.Module):
    """MLP that predicts fragility probability from hidden states + metadata."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs, dropout_rate: float = 0.1):
        self.linear1 = nnx.Linear(hidden_dim + 2, 512, rngs=rngs)
        self.linear2 = nnx.Linear(512, 128, rngs=rngs)
        self.linear3 = nnx.Linear(128, 1, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, hidden_state, layer_idx, position, *, deterministic: bool = True):
        """Forward pass.

        Args:
            hidden_state: (batch, hidden_dim) hidden state vectors (should be pre-normalized).
            layer_idx: (batch, 1) normalized layer index [0, 1].
            position: (batch, 1) normalized position [0, 1].
            deterministic: If True, disable dropout (for inference/validation).

        Returns:
            (batch, 1) predicted fragility probability [0, 1].
        """
        x = jnp.concatenate([hidden_state, layer_idx, position], axis=-1)
        x = nnx.gelu(self.linear1(x))
        x = self.dropout1(x, deterministic=deterministic)
        x = nnx.gelu(self.linear2(x))
        x = self.dropout2(x, deterministic=deterministic)
        x = self.linear3(x)
        return jax.nn.sigmoid(x)


def create_estimator(hidden_dim: int, seed: int = 0, dropout_rate: float = 0.1):
    """Create a fresh RegretEstimator with random weights."""
    rngs = nnx.Rngs(seed)
    return RegretEstimator(hidden_dim=hidden_dim, rngs=rngs, dropout_rate=dropout_rate)


def predict_regret(model: RegretEstimator, hidden_states, layer_indices,
                   positions, num_layers: int, max_seq_len: int,
                   norm_stats: dict = None):
    """Batch inference: predict fragility for multiple (hidden_state, layer, pos) tuples.

    Args:
        model: Trained RegretEstimator.
        hidden_states: (N, hidden_dim) array of hidden state vectors.
        layer_indices: (N,) array of integer layer indices.
        positions: (N,) array of integer token positions.
        num_layers: Total number of layers (for normalization).
        max_seq_len: Maximum sequence length (for normalization).
        norm_stats: Optional dict with per-layer mean/std for hidden state normalization.
                    Keys: "layer_{idx}_mean", "layer_{idx}_std" as (hidden_dim,) arrays.

    Returns:
        (N,) array of predicted fragility probabilities.
    """
    hs = jnp.array(hidden_states, dtype=jnp.float32)

    # Apply per-layer normalization if stats provided
    if norm_stats is not None:
        unique_layers = jnp.unique(layer_indices)
        for layer in unique_layers:
            layer = int(layer)
            mask = layer_indices == layer
            mean_key = f"layer_{layer}_mean"
            std_key = f"layer_{layer}_std"
            if mean_key in norm_stats and std_key in norm_stats:
                mean = jnp.array(norm_stats[mean_key])
                std = jnp.array(norm_stats[std_key])
                hs = hs.at[mask].set((hs[mask] - mean) / (std + 1e-8))

    # Normalize layer and position to [0, 1]
    layer_norm = layer_indices.astype(jnp.float32).reshape(-1, 1) / max(num_layers - 1, 1)
    pos_norm = positions.astype(jnp.float32).reshape(-1, 1) / max(max_seq_len - 1, 1)

    predictions = model(hs, layer_norm, pos_norm, deterministic=True)
    return predictions.squeeze(-1)


def save_estimator(model: RegretEstimator, path: str, norm_stats: dict = None):
    """Save estimator weights and optional normalization stats to disk."""
    state = nnx.state(model)
    flat_state = state.flat_state()
    np_dict = {}
    for path_tuple, leaf in zip(flat_state.paths, flat_state.leaves):
        key = "/".join(str(p) for p in path_tuple)
        try:
            np_dict[key] = np.array(leaf.value)
        except TypeError:
            # Skip PRNGKey arrays from dropout layers (can't convert to numpy)
            continue

    # Save normalization stats alongside weights
    if norm_stats is not None:
        for k, v in norm_stats.items():
            np_dict[f"_norm/{k}"] = np.array(v)

    np.savez(file=path, **np_dict)


def load_estimator(path: str, hidden_dim: int):
    """Load estimator weights and normalization stats from disk.

    Returns:
        (model, norm_stats) tuple. norm_stats is None if not saved.
    """
    model = create_estimator(hidden_dim)
    data = np.load(path)
    state = nnx.state(model)
    flat_state = state.flat_state()
    new_entries = []
    for path_tuple, leaf in zip(flat_state.paths, flat_state.leaves):
        key = "/".join(str(p) for p in path_tuple)
        if key in data:
            new_leaf = leaf.replace(value=jnp.array(data[key]))
            new_entries.append((path_tuple, new_leaf))
        else:
            new_entries.append((path_tuple, leaf))
    new_flat = nnx.statelib.FlatState(new_entries, sort=False)
    new_state = new_flat.to_nested_state()
    nnx.update(model, new_state)

    # Load normalization stats
    norm_stats = {}
    for key in data.files:
        if key.startswith("_norm/"):
            norm_stats[key[len("_norm/"):]] = data[key]

    return model, norm_stats if norm_stats else None
