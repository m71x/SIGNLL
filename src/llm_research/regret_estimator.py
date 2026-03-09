"""
Regret Estimator — Flax MLP Model
===================================
Lightweight MLP that predicts expected regret from a hidden state,
layer index, and token position. Designed for inference-time use
in the regret-aware early exit pipeline.

Architecture:
    Input: hidden_state (hidden_dim) + layer_idx (1) + position (1)
    → Linear(hidden_dim+2, 256) → GELU
    → Linear(256, 64) → GELU
    → Linear(64, 1)
    Output: predicted regret (scalar ≥ 0)
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


class RegretEstimator(nnx.Module):
    """MLP that predicts regret from hidden states + metadata."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(hidden_dim + 2, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, hidden_state, layer_idx, position):
        """Forward pass.

        Args:
            hidden_state: (batch, hidden_dim) hidden state vectors.
            layer_idx: (batch, 1) normalized layer index [0, 1].
            position: (batch, 1) normalized position [0, 1].

        Returns:
            (batch, 1) predicted regret values (clamped ≥ 0).
        """
        x = jnp.concatenate([hidden_state, layer_idx, position], axis=-1)
        x = nnx.gelu(self.linear1(x))
        x = nnx.gelu(self.linear2(x))
        x = self.linear3(x)
        return jax.nn.relu(x)  # regret is non-negative


def create_estimator(hidden_dim: int, seed: int = 0):
    """Create a fresh RegretEstimator with random weights."""
    rngs = nnx.Rngs(seed)
    return RegretEstimator(hidden_dim=hidden_dim, rngs=rngs)


def predict_regret(model: RegretEstimator, hidden_states, layer_indices,
                   positions, num_layers: int, max_seq_len: int):
    """Batch inference: predict regret for multiple (hidden_state, layer, pos) tuples.

    Args:
        model: Trained RegretEstimator.
        hidden_states: (N, hidden_dim) array of hidden state vectors.
        layer_indices: (N,) array of integer layer indices.
        positions: (N,) array of integer token positions.
        num_layers: Total number of layers (for normalization).
        max_seq_len: Maximum sequence length (for normalization).

    Returns:
        (N,) array of predicted regret values.
    """
    # Normalize layer and position to [0, 1]
    layer_norm = layer_indices.astype(jnp.float32).reshape(-1, 1) / max(num_layers - 1, 1)
    pos_norm = positions.astype(jnp.float32).reshape(-1, 1) / max(max_seq_len - 1, 1)

    predictions = model(hidden_states, layer_norm, pos_norm)
    return predictions.squeeze(-1)


def save_estimator(model: RegretEstimator, path: str):
    """Save estimator weights to disk."""
    state = nnx.state(model)
    flat_state = state.flat_state()
    np_dict = {}
    for path_tuple, leaf in zip(flat_state.paths, flat_state.leaves):
        key = "/".join(str(p) for p in path_tuple)
        np_dict[key] = np.array(leaf.value)
    np.savez(file=path, **np_dict)


def load_estimator(path: str, hidden_dim: int) -> RegretEstimator:
    """Load estimator weights from disk."""
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
    return model
