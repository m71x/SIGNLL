"""
Regret Estimator — Flax MLP Model
===================================
Lightweight MLP that predicts whether a token position is "fragile"
(perturbation causes regret) from hidden states, layer index, position,
and logit features (entropy, margin).

Architecture (v5 — V2 logit probing + convergence features):
    Input: normalized hidden_state (hidden_dim) + metadata features (num_meta)
           v4 num_meta=4:  [layer_idx, position, entropy, margin]
           v5 num_meta=13: [layer_idx, position, entropy, margin,
                            agreement, confidence, kl_div, int_entropy,
                            cos_sim, rel_change, proj_final,
                            is_syntax, is_whitespace]
    → Linear(hidden_dim+num_meta, 512) → LayerNorm → GELU → Dropout(0.2)
    → Linear(512, 128) → LayerNorm → GELU → Dropout(0.2)
    → Linear(128, 1) → Sigmoid
    Output: P(fragile / not converged) ∈ [0, 1]

Key changes from v4:
    - Flexible num_meta_features for V2 extra features
    - __call__ accepts optional extra_features tensor
    - Backward compatible with v4 callers (extra_features=None)
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


class RegretEstimator(nnx.Module):
    """MLP that predicts fragility/non-convergence from hidden states + metadata."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs, dropout_rate: float = 0.2,
                 num_meta_features: int = 4):
        # Total input = hidden_state (hidden_dim) + metadata (num_meta_features)
        self.num_meta_features = num_meta_features
        self.linear1 = nnx.Linear(hidden_dim + num_meta_features, 512, rngs=rngs)
        self.ln1 = nnx.LayerNorm(512, rngs=rngs)
        self.linear2 = nnx.Linear(512, 128, rngs=rngs)
        self.ln2 = nnx.LayerNorm(128, rngs=rngs)
        self.linear3 = nnx.Linear(128, 1, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, hidden_state, layer_idx, position, entropy, margin,
                 extra_features=None, *, deterministic: bool = True):
        """Forward pass.

        Args:
            hidden_state: (batch, hidden_dim) hidden state vectors (should be pre-normalized).
            layer_idx: (batch, 1) normalized layer index [0, 1].
            position: (batch, 1) normalized position [0, 1].
            entropy: (batch, 1) token entropy (z-score normalized).
            margin: (batch, 1) logit margin top1-top2 (z-score normalized).
            extra_features: (batch, V2_NUM_EXTRA_FEATURES) optional V2 features.
            deterministic: If True, disable dropout (for inference/validation).

        Returns:
            (batch, 1) predicted fragility/non-convergence probability [0, 1].
        """
        parts = [hidden_state, layer_idx, position, entropy, margin]
        if extra_features is not None:
            parts.append(extra_features)
        x = jnp.concatenate(parts, axis=-1)
        x = self.ln1(nnx.gelu(self.linear1(x)))
        x = self.dropout1(x, deterministic=deterministic)
        x = self.ln2(nnx.gelu(self.linear2(x)))
        x = self.dropout2(x, deterministic=deterministic)
        x = self.linear3(x)
        return jax.nn.sigmoid(x)


def create_estimator(hidden_dim: int, seed: int = 0, dropout_rate: float = 0.1,
                     num_meta_features: int = 4):
    """Create a fresh RegretEstimator with random weights."""
    rngs = nnx.Rngs(seed)
    return RegretEstimator(hidden_dim=hidden_dim, rngs=rngs, dropout_rate=dropout_rate,
                           num_meta_features=num_meta_features)


def predict_regret(model: RegretEstimator, hidden_states, layer_indices,
                   positions, num_layers: int, max_seq_len: int,
                   entropies=None, margins=None,
                   extra_features=None,
                   norm_stats: dict = None):
    """Batch inference: predict fragility for multiple (hidden_state, layer, pos) tuples.

    Args:
        model: Trained RegretEstimator.
        hidden_states: (N, hidden_dim) array of hidden state vectors.
        layer_indices: (N,) array of integer layer indices.
        positions: (N,) array of integer token positions.
        num_layers: Total number of layers (for normalization).
        max_seq_len: Maximum sequence length (for normalization).
        entropies: (N,) array of token entropies. Zeros if not provided.
        margins: (N,) array of logit margins. Zeros if not provided.
        extra_features: (N, V2_NUM_EXTRA_FEATURES) optional V2 features.
        norm_stats: Optional dict with per-layer mean/std for hidden state normalization.

    Returns:
        (N,) array of predicted fragility probabilities.
    """
    N = hidden_states.shape[0]
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

    # Normalize entropy and margin
    if entropies is not None:
        ent = jnp.array(entropies, dtype=jnp.float32).reshape(-1, 1)
    else:
        ent = jnp.zeros((N, 1))
    if margins is not None:
        mar = jnp.array(margins, dtype=jnp.float32).reshape(-1, 1)
    else:
        mar = jnp.zeros((N, 1))

    if norm_stats is not None:
        if "entropy_mean" in norm_stats:
            ent = (ent - norm_stats["entropy_mean"]) / (norm_stats["entropy_std"] + 1e-8)
        if "margin_mean" in norm_stats:
            mar = (mar - norm_stats["margin_mean"]) / (norm_stats["margin_std"] + 1e-8)

    # V2 extra features
    ef = None
    if extra_features is not None:
        ef = jnp.array(extra_features, dtype=jnp.float32)
        # Normalize V2 features using saved stats
        if norm_stats is not None:
            v2_feature_names = ["agreement", "confidence", "kl_div", "int_entropy",
                                "cos_sim", "rel_change", "proj_final",
                                "is_syntax", "is_whitespace"]
            for i, fname in enumerate(v2_feature_names):
                mean_key = f"v2_{fname}_mean"
                std_key = f"v2_{fname}_std"
                if mean_key in norm_stats:
                    ef = ef.at[:, i].set(
                        (ef[:, i] - norm_stats[mean_key]) / (norm_stats[std_key] + 1e-8)
                    )

    predictions = model(hs, layer_norm, pos_norm, ent, mar,
                        extra_features=ef, deterministic=True)
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


def load_estimator(path: str, hidden_dim: int, num_meta_features: int = None):
    """Load estimator weights and normalization stats from disk.

    Args:
        path: Path to .npz weights file.
        hidden_dim: Hidden state dimension.
        num_meta_features: Number of metadata features. If None, auto-detected from weights.

    Returns:
        (model, norm_stats) tuple. norm_stats is None if not saved.
    """
    data = np.load(path)

    # Auto-detect num_meta_features from linear1 input dim
    if num_meta_features is None:
        key = "linear1/kernel" if "linear1/kernel" in data else "linear1/kernel/value"
        input_dim = data[key].shape[0]
        num_meta_features = input_dim - hidden_dim

    model = create_estimator(hidden_dim, num_meta_features=num_meta_features)
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
