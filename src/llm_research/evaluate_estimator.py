"""
Regret Estimator Evaluation Suite
==================================
Pure numpy evaluation (no JAX/Flax needed). Loads weights directly from .npz.

Analyses:
  1. Prediction quality: MSE, MAE, correlation, R-squared
  2. Classification: zero vs non-zero regret (threshold sweep, AUROC)
  3. Per-layer breakdown
  4. Per-position breakdown
  5. Regret distribution analysis
  6. Early-exit simulation: compute-reward tradeoff
  7. Calibration: predicted vs actual
  8. Error analysis: worst predictions, dangerous misses
  9. Train/val split comparison
"""

import os
import sys
import numpy as np
from scipy import special  # for GELU

# Add script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (TARGET_LAYERS, TRAIN_SPLIT,
                    REGRET_DATA_PATH as _REGRET_DATA_PATH,
                    ESTIMATOR_PATH as _ESTIMATOR_PATH,
                    V2_NUM_EXTRA_FEATURES)

# ── PATHS ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
REGRET_DATA_PATH = os.path.join(PROJECT_ROOT, _REGRET_DATA_PATH)
ESTIMATOR_PATH = os.path.join(PROJECT_ROOT, f"{_ESTIMATOR_PATH}.npz")


# ── NUMPY MLP INFERENCE ─────────────────────────────────────────────
def gelu(x):
    return x * 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


V2_FEATURE_NAMES = ["agreement", "confidence", "kl_div", "int_entropy",
                     "cos_sim", "rel_change", "proj_final",
                     "is_syntax", "is_whitespace"]


class NumpyEstimator:
    """Load Flax NNX weights and run inference in pure numpy."""
    def __init__(self, path):
        data = np.load(path)
        keys = list(data.keys())
        self.weights = {}
        self.norm_stats = {}
        for k in keys:
            if k.startswith("_norm/"):
                self.norm_stats[k[len("_norm/"):]] = data[k]
            else:
                self.weights[k] = data[k]

        # Figure out the key format
        if "linear1/kernel/value" in self.weights:
            self.fmt = "value"
        elif "linear1/kernel" in self.weights:
            self.fmt = "direct"
        else:
            print(f"  Weight keys: {keys[:10]}")
            raise ValueError(f"Unknown weight format. Keys: {keys}")

        # Detect model version by layer sizes
        kernel = self._get("linear1/kernel")
        in_dim = kernel.shape[0]
        out_dim = kernel.shape[1]
        self.is_v2 = out_dim == 512  # v2+ = 512, v1 = 256
        self.has_logit_inputs = "entropy_mean" in self.norm_stats
        self.has_v2_features = "v2_agreement_mean" in self.norm_stats

        # Detect num_meta from input dim
        # hidden_dim is in_dim minus metadata features
        if self.has_v2_features:
            self.num_meta = 4 + V2_NUM_EXTRA_FEATURES  # 13
        elif self.has_logit_inputs:
            self.num_meta = 4
        else:
            self.num_meta = 2  # just layer_idx + position

        n_norm_layers = sum(1 for k in self.norm_stats if k.startswith("layer_") and k.endswith("_mean"))
        if self.norm_stats:
            print(f"  Per-layer normalization stats loaded ({n_norm_layers} layers)")
        if self.has_v2_features:
            print(f"  Model v5 detected (V2 features, {V2_NUM_EXTRA_FEATURES} extra, 512->128->1, sigmoid)")
        elif self.has_logit_inputs:
            print(f"  Model v3/v4 detected (logit features, 512->128->1, sigmoid)")
        elif self.is_v2:
            print(f"  Model v2 detected (512->128->1, sigmoid)")
        else:
            print(f"  Model v1 detected (256->64->1, ReLU)")
        print(f"  Input dim: {in_dim}, first hidden: {out_dim}")

    def _get(self, name):
        if self.fmt == "value":
            return self.weights[f"{name}/value"]
        return self.weights[name]

    def normalize_hidden_states(self, hidden_states, layer_indices):
        """Apply per-layer normalization if stats are available."""
        if not self.norm_stats:
            return hidden_states
        hs = hidden_states.copy()
        for layer in np.unique(layer_indices):
            layer = int(layer)
            mean_key = f"layer_{layer}_mean"
            std_key = f"layer_{layer}_std"
            if mean_key in self.norm_stats:
                mask = layer_indices == layer
                hs[mask] = (hs[mask] - self.norm_stats[mean_key]) / (self.norm_stats[std_key] + 1e-8)
        return hs

    def normalize_logit_features(self, entropies, margins):
        """Apply z-score normalization to entropy and margin using saved stats."""
        ent = entropies.copy()
        mar = margins.copy()
        if "entropy_mean" in self.norm_stats:
            ent = (ent - float(self.norm_stats["entropy_mean"])) / (float(self.norm_stats["entropy_std"]) + 1e-8)
        if "margin_mean" in self.norm_stats:
            mar = (mar - float(self.norm_stats["margin_mean"])) / (float(self.norm_stats["margin_std"]) + 1e-8)
        return ent, mar

    def normalize_v2_features(self, v2_features):
        """Apply z-score normalization to V2 extra features using saved stats."""
        v2f = v2_features.copy()
        for i, fname in enumerate(V2_FEATURE_NAMES):
            mean_key = f"v2_{fname}_mean"
            std_key = f"v2_{fname}_std"
            if mean_key in self.norm_stats:
                m = float(self.norm_stats[mean_key])
                s = float(self.norm_stats[std_key])
                if s > 1e-6:
                    v2f[:, i] = (v2f[:, i] - m) / (s + 1e-8)
        return v2f

    def _layer_norm(self, x, name):
        """Apply LayerNorm if weights exist."""
        scale_key = f"{name}/scale"
        bias_key = f"{name}/bias"
        try:
            scale = self._get(scale_key)
            bias = self._get(bias_key)
        except KeyError:
            return x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return scale * (x - mean) / np.sqrt(var + 1e-5) + bias

    def __call__(self, hidden_state, layer_idx, position, entropy=None, margin=None,
                 v2_features=None):
        parts = [hidden_state, layer_idx, position]
        if entropy is not None and margin is not None:
            parts.extend([entropy, margin])
        if v2_features is not None:
            parts.append(v2_features)
        x = np.concatenate(parts, axis=-1)
        x = self._layer_norm(gelu(x @ self._get("linear1/kernel") + self._get("linear1/bias")), "ln1")
        x = self._layer_norm(gelu(x @ self._get("linear2/kernel") + self._get("linear2/bias")), "ln2")
        x = x @ self._get("linear3/kernel") + self._get("linear3/bias")
        if self.is_v2:
            return sigmoid(x)
        return np.maximum(x, 0)  # v1: ReLU


def load_data():
    data = np.load(REGRET_DATA_PATH, allow_pickle=False)
    hs = data["hidden_states"]
    # Handle bfloat16 saved as raw bytes
    if hs.dtype.kind not in ('f', 'i', 'u'):
        hs_u16 = hs.view(np.uint16)
        hs_f32 = np.zeros(hs_u16.shape, dtype=np.float32)
        for i in range(hs_u16.shape[0]):
            hs_f32[i] = np.frombuffer(
                (hs_u16[i].astype(np.uint32) << 16).tobytes(), dtype=np.float32
            )
        hs = hs_f32
    result = {
        "hidden_states": np.array(hs, dtype=np.float32),
        "layer_indices": data["layer_indices"],
        "positions": data["positions"],
        "regret_values": data["regret_values"],
    }
    # Load logit features if available
    if "entropies" in data:
        result["entropies"] = data["entropies"].astype(np.float32)
        result["margins"] = data["margins"].astype(np.float32)
    # Load V2 features if available
    if "v2_features" in data:
        result["v2_features"] = data["v2_features"].astype(np.float32)
    return result


def predict_all(model, data):
    hs = data["hidden_states"].copy()
    layers = data["layer_indices"]
    positions = data["positions"]

    # Apply per-layer normalization
    hs = model.normalize_hidden_states(hs, layers)

    num_layers = max(int(np.max(layers)) + 1, 1)
    max_pos = max(int(np.max(positions)) + 1, 1)
    layer_norm = layers.astype(np.float32).reshape(-1, 1) / (num_layers - 1)
    pos_norm = positions.astype(np.float32).reshape(-1, 1) / (max_pos - 1)

    # Use logit features if available
    ent, mar, v2f = None, None, None
    if "entropies" in data:
        ent = data["entropies"].reshape(-1, 1)
        mar = data["margins"].reshape(-1, 1)
        ent, mar = model.normalize_logit_features(ent, mar)

    # Use V2 features if available
    if "v2_features" in data and model.has_v2_features:
        v2f = model.normalize_v2_features(data["v2_features"])

    if ent is not None:
        preds = model(hs, layer_norm, pos_norm, ent, mar, v2_features=v2f)
    else:
        preds = model(hs, layer_norm, pos_norm)
    return preds.squeeze(-1)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def eval_prediction_quality(preds, targets):
    section("1. PREDICTION QUALITY")

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    corr = np.corrcoef(preds, targets)[0, 1] if np.std(preds) > 0 and np.std(targets) > 0 else 0.0

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    print(f"  MSE:         {mse:.6f}")
    print(f"  RMSE:        {rmse:.6f}")
    print(f"  MAE:         {mae:.6f}")
    print(f"  Pearson r:   {corr:.4f}")
    print(f"  R-squared:   {r2:.4f}")

    print(f"\n  Target stats:     mean={np.mean(targets):.4f}, std={np.std(targets):.4f}, "
          f"min={np.min(targets):.4f}, max={np.max(targets):.4f}")
    print(f"  Prediction stats: mean={np.mean(preds):.4f}, std={np.std(preds):.4f}, "
          f"min={np.min(preds):.4f}, max={np.max(preds):.4f}")

    return {"mse": mse, "mae": mae, "corr": corr, "r2": r2}


def eval_classification(preds, targets):
    section("2. CLASSIFICATION: ZERO vs NON-ZERO REGRET")

    y_true = (targets > 0).astype(int)
    n_pos = np.sum(y_true)
    n_neg = np.sum(1 - y_true)
    print(f"  Positive (regret > 0): {n_pos} ({100*n_pos/len(y_true):.1f}%)")
    print(f"  Negative (regret = 0): {n_neg} ({100*n_neg/len(y_true):.1f}%)")

    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'─'*50}")

    best_f1 = 0
    best_thresh = 0
    for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        y_pred = (preds > thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        accuracy = (tp + tn) / len(y_true)

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            marker = " <-- best F1"

        print(f"  {thresh:>10.2f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {accuracy:>10.3f}{marker}")

    # AUROC
    thresholds = np.linspace(0, np.max(preds) + 0.01, 200)
    tprs, fprs = [], []
    for t in thresholds:
        y_pred = (preds >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))
    sorted_pairs = sorted(zip(fprs, tprs))
    fprs_s = [p[0] for p in sorted_pairs]
    tprs_s = [p[1] for p in sorted_pairs]
    auroc = np.trapezoid(tprs_s, fprs_s)
    print(f"\n  AUROC: {auroc:.4f}")
    print(f"  Best F1: {best_f1:.4f} at threshold={best_thresh:.2f}")

    return {"auroc": auroc, "best_f1": best_f1, "best_thresh": best_thresh}


def eval_per_layer(preds, targets, layer_indices):
    section("3. PER-LAYER BREAKDOWN")

    unique_layers = sorted(np.unique(layer_indices))
    print(f"\n  {'Layer':>6} {'N':>6} {'MSE':>10} {'MAE':>10} {'Corr':>10} {'Mean Tgt':>10} {'Mean Pred':>10}")
    print(f"  {'─'*62}")

    for layer in unique_layers:
        mask = layer_indices == layer
        p = preds[mask]
        t = targets[mask]
        mse = np.mean((p - t) ** 2)
        mae = np.mean(np.abs(p - t))
        corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 0 and np.std(t) > 0 else 0.0
        print(f"  {layer:>6} {np.sum(mask):>6} {mse:>10.6f} {mae:>10.6f} "
              f"{corr:>10.4f} {np.mean(t):>10.4f} {np.mean(p):>10.4f}")


def eval_per_position(preds, targets, positions):
    section("4. PER-POSITION BREAKDOWN (quintiles)")

    percentiles = np.percentile(positions, [0, 20, 40, 60, 80, 100])
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    print(f"\n  {'Bin':>10} {'Pos Range':>12} {'N':>6} {'MSE':>10} {'Mean Tgt':>10} {'Mean Pred':>10} {'NZ%':>8}")
    print(f"  {'─'*68}")

    for i in range(len(bin_labels)):
        lo, hi = percentiles[i], percentiles[i + 1]
        mask = (positions >= lo) & (positions <= hi) if i == len(bin_labels) - 1 else (positions >= lo) & (positions < hi)
        if np.sum(mask) == 0:
            continue
        p = preds[mask]
        t = targets[mask]
        mse = np.mean((p - t) ** 2)
        nz = 100 * np.mean(t > 0)
        print(f"  {bin_labels[i]:>10} {lo:>5.0f}-{hi:>5.0f} {np.sum(mask):>6} "
              f"{mse:>10.6f} {np.mean(t):>10.4f} {np.mean(p):>10.4f} {nz:>7.1f}%")


def eval_regret_distribution(targets, is_v2=False):
    section("5. TARGET DISTRIBUTION" if is_v2 else "5. REGRET DISTRIBUTION")

    print(f"  Total samples:    {len(targets)}")
    if is_v2:
        print(f"  Converged (< 0.1):   {np.sum(targets < 0.1)} ({100*np.mean(targets < 0.1):.1f}%)")
        print(f"  Divergent (>= 0.5):  {np.sum(targets >= 0.5)} ({100*np.mean(targets >= 0.5):.1f}%)")
        print(f"  Mean: {np.mean(targets):.4f}, Std: {np.std(targets):.4f}")

        # Histogram of continuous target values
        print(f"\n  Target value histogram:")
        bins = np.linspace(0, 1, 11)
        counts, _ = np.histogram(targets, bins=bins)
        for i, count in enumerate(counts):
            bar = '#' * int(40 * count / max(max(counts), 1))
            print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:>5} {bar}")
    else:
        print(f"  Zero regret:      {np.sum(targets == 0)} ({100*np.mean(targets == 0):.1f}%)")
        print(f"  Non-zero regret:  {np.sum(targets > 0)} ({100*np.mean(targets > 0):.1f}%)")

        nonzero = targets[targets > 0]
        if len(nonzero) > 0:
            print(f"\n  Non-zero regret histogram:")
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
            counts, _ = np.histogram(nonzero, bins=bins)
            for i, count in enumerate(counts):
                bar = '#' * int(40 * count / max(max(counts), 1))
                print(f"    ({bins[i]:.1f}, {bins[i+1]:.1f}]: {count:>5} {bar}")

        unique_vals = np.unique(np.round(targets, 4))
        if len(unique_vals) <= 30:
            print(f"\n  Unique regret values ({len(unique_vals)}):")
            for v in unique_vals:
                c = np.sum(np.abs(targets - v) < 1e-4)
                pct = 100 * c / len(targets)
                print(f"    {v:.4f}: {c:>5} ({pct:.1f}%)")
        else:
            print(f"\n  {len(unique_vals)} unique regret values (continuous distribution)")


def eval_early_exit_simulation(preds, targets, layer_indices, is_v2=False):
    section("6. EARLY-EXIT SIMULATION")
    if is_v2:
        print("  V2: prediction = P(not converged). Exit when prediction < threshold.")
    else:
        print("  At each layer (early->late), if predicted regret < threshold -> exit early")
    print("  'Safe exit' = true target was 0 (converged/safe), 'Unsafe exit' = true target > 0")

    n_layers = len(TARGET_LAYERS)

    # Determine perturbation count (excluding appended failing baselines)
    perturb_path = os.path.join(PROJECT_ROOT, "phase2a_perturbations.json")
    if os.path.exists(perturb_path):
        import json
        with open(perturb_path) as f:
            n_perturbations = len(json.load(f))
        # Only use perturbation samples for early-exit simulation
        n_pert_samples = n_perturbations * n_layers
        preds = preds[:n_pert_samples]
        targets = targets[:n_pert_samples]
    else:
        n_perturbations = len(targets) // n_layers

    print(f"\n  Perturbations: {n_perturbations}, Layers: {TARGET_LAYERS}")

    # V2: use different safe/unsafe threshold since targets are continuous
    safe_thresh = 0.1 if is_v2 else 0.0

    print(f"\n  {'Thresh':>8} {'ExitRate':>9} {'SafeExit%':>10} {'UnsafeExit%':>12} "
          f"{'AvgLayer':>9} {'Compute%':>9}")
    print(f"  {'─'*58}")

    max_layer = TARGET_LAYERS[-1]
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        exits = 0
        safe_exits = 0
        unsafe_exits = 0
        total_layers_used = 0

        for p in range(n_perturbations):
            exited = False
            for l_offset in range(n_layers):
                idx = p * n_layers + l_offset
                if preds[idx] < thresh:
                    exits += 1
                    total_layers_used += TARGET_LAYERS[l_offset]
                    if targets[idx] <= safe_thresh:
                        safe_exits += 1
                    else:
                        unsafe_exits += 1
                    exited = True
                    break
            if not exited:
                total_layers_used += max_layer

        exit_rate = exits / n_perturbations
        safe_pct = 100 * safe_exits / max(exits, 1)
        unsafe_pct = 100 * unsafe_exits / max(exits, 1)
        avg_layers = total_layers_used / n_perturbations
        compute_pct = 100 * avg_layers / max_layer

        print(f"  {thresh:>8.2f} {exit_rate:>8.1%} {safe_pct:>9.1f}% {unsafe_pct:>11.1f}% "
              f"{avg_layers:>9.1f} {compute_pct:>8.1f}%")


def eval_calibration(preds, targets):
    section("7. CALIBRATION: PREDICTED vs ACTUAL")

    n_bins = 10
    pred_min, pred_max = np.min(preds), np.max(preds)
    pred_bins = np.linspace(pred_min, pred_max + 1e-6, n_bins + 1)

    print(f"\n  {'Pred Bin':>15} {'N':>6} {'Mean Pred':>10} {'Mean Actual':>12} {'Gap':>10}")
    print(f"  {'─'*54}")

    for i in range(n_bins):
        mask = (preds >= pred_bins[i]) & (preds < pred_bins[i + 1])
        if np.sum(mask) == 0:
            continue
        mean_pred = np.mean(preds[mask])
        mean_actual = np.mean(targets[mask])
        gap = mean_pred - mean_actual
        print(f"  [{pred_bins[i]:>5.3f},{pred_bins[i+1]:>6.3f}) {np.sum(mask):>6} "
              f"{mean_pred:>10.4f} {mean_actual:>12.4f} {gap:>+10.4f}")


def eval_error_analysis(preds, targets, layer_indices, positions):
    section("8. ERROR ANALYSIS")

    errors = np.abs(preds - targets)

    # Worst predictions
    n = 15
    worst_idx = np.argsort(errors)[-n:][::-1]
    print(f"\n  Top {n} worst predictions:")
    print(f"  {'Idx':>6} {'Layer':>6} {'Pos':>6} {'Pred':>10} {'Actual':>10} {'Error':>10}")
    print(f"  {'─'*50}")
    for idx in worst_idx:
        print(f"  {idx:>6} {layer_indices[idx]:>6} {positions[idx]:>6} "
              f"{preds[idx]:>10.4f} {targets[idx]:>10.4f} {errors[idx]:>10.4f}")

    # Dangerous misses
    print(f"\n  DANGEROUS MISSES (high target predicted as safe):")
    for regret_thresh, pred_thresh in [(0.5, 0.1), (0.3, 0.1), (0.5, 0.2)]:
        high_regret = targets > regret_thresh
        low_pred = preds < pred_thresh
        dangerous = high_regret & low_pred
        n_d = np.sum(dangerous)
        print(f"    target>{regret_thresh} but pred<{pred_thresh}: "
              f"{n_d}/{np.sum(high_regret)} ({100*n_d/max(np.sum(high_regret),1):.1f}%)")

    # False alarms
    print(f"\n  FALSE ALARMS (zero/low target predicted as risky):")
    for pred_thresh in [0.3, 0.5, 0.7]:
        zero_regret = targets < 0.05  # near-zero target
        high_pred = preds > pred_thresh
        false_alarm = zero_regret & high_pred
        n_f = np.sum(false_alarm)
        print(f"    target<0.05 but pred>{pred_thresh}: "
              f"{n_f}/{np.sum(zero_regret)} ({100*n_f/max(np.sum(zero_regret),1):.1f}%)")


def eval_v2_features(data):
    """V2-specific: analyze the extra features and their correlation with targets."""
    if "v2_features" not in data:
        return

    section("10. V2 FEATURE ANALYSIS")

    v2f = data["v2_features"]
    targets = data["regret_values"]

    print(f"\n  {'Feature':>16} {'Mean':>10} {'Std':>10} {'Corr w/ Target':>15}")
    print(f"  {'─'*52}")

    for i, fname in enumerate(V2_FEATURE_NAMES):
        col = v2f[:, i]
        corr = np.corrcoef(col, targets)[0, 1] if np.std(col) > 0 else 0.0
        print(f"  {fname:>16} {np.mean(col):>10.4f} {np.std(col):>10.4f} {corr:>15.4f}")

    # Per-layer agreement rates
    layers = data["layer_indices"]
    unique_layers = sorted(np.unique(layers))
    agreement_col = v2f[:, 0]  # agreement is first V2 feature

    print(f"\n  Per-layer agreement rate (argmax match with final layer):")
    print(f"  {'Layer':>6} {'Agreement%':>12} {'Mean KL Target':>15}")
    print(f"  {'─'*35}")
    for layer in unique_layers:
        mask = layers == layer
        agree_rate = 100 * np.mean(agreement_col[mask])
        mean_tgt = np.mean(targets[mask])
        print(f"  {layer:>6} {agree_rate:>11.1f}% {mean_tgt:>15.4f}")


def eval_train_val(model, data):
    section("9. TRAIN vs VALIDATION SPLIT")

    N = len(data["regret_values"])
    targets = data["regret_values"]
    is_v2 = "v2_features" in data
    binary = (targets > (0.5 if is_v2 else 0)).astype(float)

    # Stratified split matching training code
    perm = np.random.RandomState(42).permutation(N)
    split_thresh = 0.5 if is_v2 else 0
    fragile_idx = perm[targets[perm] > split_thresh]
    safe_idx = perm[targets[perm] <= split_thresh]
    f_split = int(len(fragile_idx) * TRAIN_SPLIT)
    s_split = int(len(safe_idx) * TRAIN_SPLIT)
    train_idx = np.concatenate([fragile_idx[:f_split], safe_idx[:s_split]])
    val_idx = np.concatenate([fragile_idx[f_split:], safe_idx[s_split:]])

    hs = model.normalize_hidden_states(data["hidden_states"].copy(), data["layer_indices"])
    layers = data["layer_indices"]
    positions = data["positions"]

    num_layers = max(int(np.max(layers)) + 1, 1)
    max_pos = max(int(np.max(positions)) + 1, 1)
    layer_norm = layers.astype(np.float32).reshape(-1, 1) / (num_layers - 1)
    pos_norm = positions.astype(np.float32).reshape(-1, 1) / (max_pos - 1)

    print(f"\n  {'Split':>6} {'N':>6} {'MSE':>10} {'MAE':>10} {'Corr':>10} {'Acc':>8}")
    print(f"  {'─'*52}")

    # Normalize logit features if available
    has_logit = "entropies" in data
    if has_logit:
        ent = data["entropies"].reshape(-1, 1)
        mar = data["margins"].reshape(-1, 1)
        ent, mar = model.normalize_logit_features(ent, mar)

    # Normalize V2 features if available
    v2f = None
    if "v2_features" in data and model.has_v2_features:
        v2f = model.normalize_v2_features(data["v2_features"])

    for name, idx in [("Train", train_idx), ("Val", val_idx)]:
        v2f_batch = v2f[idx] if v2f is not None else None
        if has_logit:
            p = model(hs[idx], layer_norm[idx], pos_norm[idx], ent[idx], mar[idx],
                      v2_features=v2f_batch).squeeze(-1)
        else:
            p = model(hs[idx], layer_norm[idx], pos_norm[idx]).squeeze(-1)
        t = targets[idx]
        b = binary[idx]
        mse = np.mean((p - t) ** 2)
        mae = np.mean(np.abs(p - t))
        corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 0 and np.std(t) > 0 else 0.0
        acc = np.mean((p > 0.5) == b)
        print(f"  {name:>6} {len(idx):>6} {mse:>10.6f} {mae:>10.6f} {corr:>10.4f} {acc:>7.3f}")


def print_improvements(preds, targets):
    section("11. IMPROVEMENT SUGGESTIONS")

    corr = np.corrcoef(preds, targets)[0, 1] if np.std(preds) > 0 else 0
    pred_std = np.std(preds)
    tgt_std = np.std(targets)

    suggestions = []

    if corr < 0.3:
        suggestions.append(
            "LOW CORRELATION (r={:.3f}): MLP struggles with target prediction.\n"
            "  -> Deeper network: add residual connections.\n"
            "  -> Consider attention-based architecture for layer interactions.".format(corr)
        )

    if np.mean(preds) > np.mean(targets) * 1.5:
        suggestions.append(
            "OVER-PREDICTION: mean pred={:.3f} vs mean target={:.3f}.\n"
            "  -> Reduce class weight or tune pos_weight.\n"
            "  -> Add asymmetric loss: penalize over-prediction less.".format(
                np.mean(preds), np.mean(targets))
        )

    if pred_std < tgt_std * 0.3:
        suggestions.append(
            "LOW VARIANCE: pred_std={:.4f} vs target_std={:.4f}.\n"
            "  -> Model may be underfitting. Add layers or increase capacity.".format(pred_std, tgt_std)
        )

    suggestions.append(
        "NEXT STEPS:\n"
        "  -> End-to-end early-exit test: run model with regret gating on held-out problems.\n"
        "  -> Compare against confidence-only baseline (entropy/margin threshold).\n"
        "  -> Measure actual compute savings vs quality degradation curve.\n"
        "  -> Cross-validate: retrain on different random seeds and check stability."
    )

    for i, s in enumerate(suggestions, 1):
        print(f"\n  {i}. {s}")


def main():
    print("=" * 70)
    print("  REGRET ESTIMATOR EVALUATION SUITE")
    print("=" * 70)

    print("\nLoading data...")
    data = load_data()
    N = len(data["regret_values"])
    hdim = data["hidden_states"].shape[1]
    is_v2 = "v2_features" in data
    print(f"  Samples: {N}, Hidden dim: {hdim}")
    print(f"  Layers: {sorted(np.unique(data['layer_indices']))}")
    if "entropies" in data:
        print(f"  Logit features: entropy [{data['entropies'].min():.2f}, {data['entropies'].max():.2f}], "
              f"margin [{data['margins'].min():.2f}, {data['margins'].max():.2f}]")
    if is_v2:
        print(f"  V2 features: {data['v2_features'].shape[1]} extra features")
        print(f"  V2 KL targets: mean={data['regret_values'].mean():.4f}, "
              f"std={data['regret_values'].std():.4f}")

    print("Loading model...")
    model = NumpyEstimator(ESTIMATOR_PATH)
    print(f"  Loaded from {ESTIMATOR_PATH}")

    print("Running inference...")
    preds = predict_all(model, data)
    targets = data["regret_values"]
    layers = data["layer_indices"]
    positions = data["positions"]
    print(f"  Done. Predictions shape: {preds.shape}")

    eval_prediction_quality(preds, targets)
    eval_classification(preds, targets)
    eval_per_layer(preds, targets, layers)
    eval_per_position(preds, targets, positions)
    eval_regret_distribution(targets, is_v2=is_v2)
    eval_early_exit_simulation(preds, targets, layers, is_v2=is_v2)
    eval_calibration(preds, targets)
    eval_error_analysis(preds, targets, layers, positions)
    eval_train_val(model, data)
    if is_v2:
        eval_v2_features(data)
    print_improvements(preds, targets)

    print(f"\n{'='*70}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
