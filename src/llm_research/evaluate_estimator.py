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
import numpy as np
from scipy import special  # for GELU

# ── PATHS ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
REGRET_DATA_PATH = os.path.join(PROJECT_ROOT, "regret_dataset.npz")
ESTIMATOR_PATH = os.path.join(PROJECT_ROOT, "regret_estimator_weights.npz")

TARGET_LAYERS = [8, 24, 40]


# ── NUMPY MLP INFERENCE ─────────────────────────────────────────────
def gelu(x):
    return x * 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


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

        # Detect model version by hidden size
        kernel = self._get("linear1/kernel")
        out_dim = kernel.shape[1]
        self.is_v2 = out_dim == 512  # v2 = 512, v1 = 256

        if self.norm_stats:
            print(f"  Per-layer normalization stats loaded ({len(self.norm_stats)//2} layers)")
        if self.is_v2:
            print(f"  Model v2 detected (512→128→1, sigmoid)")
        else:
            print(f"  Model v1 detected (256→64→1, ReLU)")

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

    def __call__(self, hidden_state, layer_idx, position):
        x = np.concatenate([hidden_state, layer_idx, position], axis=-1)
        x = gelu(x @ self._get("linear1/kernel") + self._get("linear1/bias"))
        x = gelu(x @ self._get("linear2/kernel") + self._get("linear2/bias"))
        x = x @ self._get("linear3/kernel") + self._get("linear3/bias")
        if self.is_v2:
            return sigmoid(x)
        return np.maximum(x, 0)  # v1: ReLU


def load_data():
    data = np.load(REGRET_DATA_PATH, allow_pickle=False)
    hs = data["hidden_states"]
    # Handle bfloat16 saved as raw bytes
    if hs.dtype.kind not in ('f', 'i', 'u'):
        # bfloat16 as uint16 → float32 conversion
        hs_u16 = hs.view(np.uint16)
        # bfloat16 to float32: shift left 16 bits
        hs_f32 = np.zeros(hs_u16.shape, dtype=np.float32)
        for i in range(hs_u16.shape[0]):
            hs_f32[i] = np.frombuffer(
                (hs_u16[i].astype(np.uint32) << 16).tobytes(), dtype=np.float32
            )
        hs = hs_f32
    return {
        "hidden_states": np.array(hs, dtype=np.float32),
        "layer_indices": data["layer_indices"],
        "positions": data["positions"],
        "regret_values": data["regret_values"],
    }


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


def eval_regret_distribution(targets):
    section("5. REGRET DISTRIBUTION")

    print(f"  Total samples:    {len(targets)}")
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
            bar = '#' * int(30 * c / len(targets) * (100 / max(np.max(np.bincount(np.round(targets * 100).astype(int))), 1)))
            print(f"    {v:.4f}: {c:>5} ({pct:.1f}%)")
    else:
        print(f"\n  {len(unique_vals)} unique regret values (continuous distribution)")


def eval_early_exit_simulation(preds, targets, layer_indices):
    section("6. EARLY-EXIT SIMULATION")
    print("  At each layer (early->late), if predicted regret < threshold -> exit early")
    print("  'Safe exit' = true regret was 0, 'Unsafe exit' = true regret > 0")

    n_layers = len(TARGET_LAYERS)
    n_perturbations = len(targets) // n_layers

    print(f"\n  Perturbations: {n_perturbations}, Layers: {TARGET_LAYERS}")

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
                    if targets[idx] == 0:
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
    section("7. CALIBRATION: PREDICTED vs ACTUAL REGRET")

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
    print(f"\n  DANGEROUS MISSES (high regret predicted as safe):")
    for regret_thresh, pred_thresh in [(0.5, 0.1), (0.3, 0.1), (0.5, 0.2)]:
        high_regret = targets > regret_thresh
        low_pred = preds < pred_thresh
        dangerous = high_regret & low_pred
        n_d = np.sum(dangerous)
        print(f"    regret>{regret_thresh} but pred<{pred_thresh}: "
              f"{n_d}/{np.sum(high_regret)} ({100*n_d/max(np.sum(high_regret),1):.1f}%)")

    # False alarms
    print(f"\n  FALSE ALARMS (zero regret predicted as risky):")
    for pred_thresh in [0.3, 0.5, 0.7]:
        zero_regret = targets == 0
        high_pred = preds > pred_thresh
        false_alarm = zero_regret & high_pred
        n_f = np.sum(false_alarm)
        print(f"    regret=0 but pred>{pred_thresh}: "
              f"{n_f}/{np.sum(zero_regret)} ({100*n_f/max(np.sum(zero_regret),1):.1f}%)")


def eval_train_val(model, data):
    section("9. TRAIN vs VALIDATION SPLIT")

    N = len(data["regret_values"])
    targets = data["regret_values"]
    binary = (targets > 0).astype(float)

    # Stratified split matching training code
    perm = np.random.RandomState(42).permutation(N)
    fragile_idx = perm[binary[perm] > 0]
    safe_idx = perm[binary[perm] == 0]
    f_split = int(len(fragile_idx) * 0.8)
    s_split = int(len(safe_idx) * 0.8)
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

    for name, idx in [("Train", train_idx), ("Val", val_idx)]:
        p = model(hs[idx], layer_norm[idx], pos_norm[idx]).squeeze(-1)
        t = targets[idx]
        b = binary[idx]
        mse = np.mean((p - t) ** 2)
        mae = np.mean(np.abs(p - t))
        corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 0 and np.std(t) > 0 else 0.0
        acc = np.mean((p > 0.5) == b)
        print(f"  {name:>6} {len(idx):>6} {mse:>10.6f} {mae:>10.6f} {corr:>10.4f} {acc:>7.3f}")


def print_improvements(preds, targets):
    section("10. IMPROVEMENT SUGGESTIONS")

    corr = np.corrcoef(preds, targets)[0, 1] if np.std(preds) > 0 else 0
    unique_regrets = len(np.unique(np.round(targets, 2)))
    pred_std = np.std(preds)
    tgt_std = np.std(targets)

    suggestions = []

    if corr < 0.3:
        suggestions.append(
            "LOW CORRELATION (r={:.3f}): MLP struggles with regret magnitude.\n"
            "  -> Deeper network: add 512-unit layer, or use residual connections.\n"
            "  -> Add input features: token entropy, logit margin from forward pass.".format(corr)
        )

    if unique_regrets < 20:
        suggestions.append(
            "COARSE REGRET ({} unique values): partial reward may be too discrete.\n"
            "  -> Weight test assertions by difficulty/coverage.\n"
            "  -> Use log-probability regret: KL divergence between baseline & perturbed.\n"
            "  -> Run multiple rollouts per perturbation (3-5) and average.".format(unique_regrets)
        )

    if np.mean(preds) > np.mean(targets) * 1.5:
        suggestions.append(
            "OVER-PREDICTION: mean pred={:.3f} vs mean target={:.3f}.\n"
            "  -> Reduce focal alpha from 10.0 to 3.0-5.0.\n"
            "  -> Add asymmetric loss: penalize over-prediction less than under-prediction.".format(
                np.mean(preds), np.mean(targets))
        )

    if pred_std < tgt_std * 0.3:
        suggestions.append(
            "LOW VARIANCE: pred_std={:.4f} vs target_std={:.4f}.\n"
            "  -> Model may be underfitting. Increase hidden sizes or add layers.\n"
            "  -> Try softplus output instead of ReLU (smoother gradients near 0).".format(pred_std, tgt_std)
        )

    suggestions.append(
        "DATA SCALE:\n"
        "  -> Current: 5 positions/prompt, 3 layers, ~310 prompts = 4650 samples.\n"
        "  -> Increase to 10 positions/prompt -> ~9300 samples.\n"
        "  -> Add layers [4, 12, 16, 20, 28, 32, 36, 44] for finer granularity.\n"
        "  -> Multiple perturbations per position (different competitor tokens)."
    )

    suggestions.append(
        "ARCHITECTURE:\n"
        "  -> Learnable layer embeddings instead of scalar index.\n"
        "  -> Add dropout (0.1-0.2) for regularization.\n"
        "  -> Try binary classification (fragile/safe) as auxiliary task.\n"
        "  -> Wider bottleneck: 256->128->1 instead of 256->64->1."
    )

    suggestions.append(
        "TRAINING:\n"
        "  -> Cosine LR schedule with warmup (currently flat 1e-4).\n"
        "  -> Train 200+ epochs with patience-based early stopping.\n"
        "  -> Try Huber loss instead of MSE (robust to outliers).\n"
        "  -> Stratified train/val split to balance regret distribution."
    )

    suggestions.append(
        "EVALUATION (next steps):\n"
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
    print(f"  Samples: {N}, Hidden dim: {hdim}")
    print(f"  Layers: {sorted(np.unique(data['layer_indices']))}")

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
    eval_regret_distribution(targets)
    eval_early_exit_simulation(preds, targets, layers)
    eval_calibration(preds, targets)
    eval_error_analysis(preds, targets, layers, positions)
    eval_train_val(model, data)
    print_improvements(preds, targets)

    print(f"\n{'='*70}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
