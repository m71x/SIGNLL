"""Prep script: preserve Phase 2b regret values, clear Phase 2a for re-run.

Run on master worker (worker 2) before deploying with new TARGET_LAYERS.
Extracts per-perturbation regret values from regret_dataset.npz and saves
as phase2b_checkpoint.json so Phase 2b skips rollouts on next run.

If --full is passed, also clears Phase 2b checkpoint (for when
MAX_POSITIONS_PER_PROMPT changed and perturbations are incompatible).
"""
import json
import numpy as np
import os
import sys

REGRET_DATA_PATH = "regret_dataset.npz"
PHASE2A_HIDDEN_PATH = "phase2a_hidden_states.npz"
PHASE2A_PERTURB_PATH = "phase2a_perturbations.json"
PHASE2A_FAILING_PATH = "phase2a_failing_hidden.npz"
PHASE2B_CHECKPOINT_PATH = "phase2b_checkpoint.json"

full_reset = "--full" in sys.argv

if full_reset:
    print("FULL RESET: clearing all Phase 2a, 2b, and dataset files")
    for path in [PHASE2A_HIDDEN_PATH, PHASE2A_PERTURB_PATH, PHASE2A_FAILING_PATH,
                 REGRET_DATA_PATH, PHASE2B_CHECKPOINT_PATH]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted {path}")
    print("\nReady for full re-run (Phase 2a + 2b + 3).")
    sys.exit(0)

# Normal mode: preserve Phase 2b rollouts, clear Phase 2a
if not os.path.exists(REGRET_DATA_PATH):
    print("No regret_dataset.npz found — nothing to preserve")
    sys.exit(1)

# Load existing regret dataset
data = np.load(REGRET_DATA_PATH)
regret_all = data["regret_values"]
layer_indices = data["layer_indices"]

# Load perturbation metadata to get num_layers
with open(PHASE2A_PERTURB_PATH) as f:
    perturbations = json.load(f)

old_num_layers = perturbations[0]["num_layers"]
n_perturbations = len(perturbations)
print(f"Found {n_perturbations} perturbations, {old_num_layers} layers each")
print(f"Total hidden states: {len(regret_all)} ({n_perturbations} x {old_num_layers})")

# Extract per-perturbation regret (take from first layer of each group, they're all the same)
regret_per_pert = []
for i in range(n_perturbations):
    regret_per_pert.append(float(regret_all[i * old_num_layers]))

# Verify they're consistent across layers
for i in range(n_perturbations):
    for j in range(old_num_layers):
        assert regret_all[i * old_num_layers + j] == regret_per_pert[i], \
            f"Regret mismatch at perturbation {i}, layer offset {j}"
print("All regret values consistent across layers")

# Save as Phase 2b checkpoint
checkpoint = {
    "completed": n_perturbations,
    "regret_values": regret_per_pert,
}
with open(PHASE2B_CHECKPOINT_PATH, "w") as f:
    json.dump(checkpoint, f)
print(f"Saved phase2b checkpoint: {n_perturbations} completed rollouts")

nonzero = sum(1 for v in regret_per_pert if v > 0)
print(f"Non-zero regret: {nonzero}/{n_perturbations} ({100*nonzero/n_perturbations:.1f}%)")

# Delete Phase 2a outputs and regret dataset (will be regenerated with new layers)
for path in [PHASE2A_HIDDEN_PATH, PHASE2A_PERTURB_PATH, PHASE2A_FAILING_PATH, REGRET_DATA_PATH]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted {path}")

print("\nReady for re-run. Phase 2a will re-run with new TARGET_LAYERS.")
print("Phase 2b will load checkpoint and skip rollouts.")
