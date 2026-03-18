"""
Shared Configuration — Regret-Aware Early Exit Pipeline
=========================================================
Central config for constants shared across training, evaluation, and prep scripts.
"""

# Model
MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"

# Layer checkpoints for hidden state extraction and early-exit simulation
TARGET_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

# Perturbation parameters
PERTURB_STRENGTH = 2.0
PERTURB_TEMP = 0.3
PERTURB_TOP_K = 3
MAX_POSITIONS_PER_PROMPT = 10

# Generation
MAX_GEN_TOKENS = 512
ROLLOUT_MAX_TOKENS = 128
CODE_TIMEOUT = 10

# Regret estimator training
ESTIMATOR_LR = 3e-4
ESTIMATOR_EPOCHS = 200
ESTIMATOR_BATCH = 256
ESTIMATOR_PATIENCE = 30
ESTIMATOR_DROPOUT = 0.2
ESTIMATOR_WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
TRAIN_SPLIT = 0.8

# Output paths
BASELINE_PATH = "regret_baseline_results.json"
PHASE2A_HIDDEN_PATH = "phase2a_hidden_states.npz"
PHASE2A_PERTURB_PATH = "phase2a_perturbations.json"
PHASE2A_FAILING_PATH = "phase2a_failing_hidden.npz"
REGRET_DATA_PATH = "regret_dataset.npz"
ESTIMATOR_PATH = "regret_estimator_weights"
PHASE2B_CHECKPOINT_PATH = "phase2b_checkpoint.json"
PHASE2B_CHECKPOINT_INTERVAL = 25

# Batch processing
BATCH_SIZE = 4

# ── V2: Logit Probing Mode ─────────────────────────────────────────
# When V2_MODE is True, Phase 2a computes layer-dependent targets via
# intermediate logit probing + convergence metrics + token-type features.
# Phase 2b (rollouts) is skipped and training uses logit agreement targets.
V2_MODE = True
V2_SKIP_PHASE2B = True  # Skip Phase 2b rollouts (train on logit targets only)

# V2 extra input features (per layer per position):
#   0: agreement       — binary: argmax(intermediate) == argmax(final)
#   1: confidence      — softmax prob of top token at intermediate layer
#   2: kl_div          — KL(final || intermediate), log-scaled
#   3: int_entropy     — entropy of intermediate logit distribution
#   4: cos_sim         — cosine similarity to previous layer (log-scaled)
#   5: rel_change      — relative L2 change from previous layer (log-scaled)
#   6: proj_final      — projection onto final hidden state direction
#   7: is_syntax       — 1.0 if token is syntax/boilerplate
#   8: is_whitespace   — 1.0 if token is whitespace/indentation
V2_NUM_EXTRA_FEATURES = 9

# Syntax/boilerplate tokens (safe to exit early)
SYNTAX_TOKENS = frozenset({
    'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'in', 'not',
    'and', 'or', 'is', 'None', 'True', 'False', 'class', 'import', 'from',
    'try', 'except', 'finally', 'with', 'as', 'pass', 'break', 'continue',
    'lambda', 'yield', 'raise', 'del', 'global', 'nonlocal', 'assert',
    ':', '(', ')', '[', ']', '{', '}', ',', '.', '=', '==', '!=',
    '+=', '-=', '*=', '/=', '->', '**', '//', '%', '<', '>', '<=', '>=',
    '+', '-', '*', '/', '&', '|', '^', '~', '@', ';',
})
