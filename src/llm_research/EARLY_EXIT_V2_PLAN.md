Markdown# Early Exit V2: Strategy Overhaul for Significantly Earlier Exits

## The Core Problem

The current system achieves an AUROC of 0.982 for fragility classification but exits at an average layer of **38.6/44** — saving only ~12% of compute. This happens because of a fundamental design flaw:

**Regret is position-level, not layer-level.** When a perturbation at position P causes regret, that same regret value is assigned to ALL 11 target layers. The model has no layer-dependent signal — it can't learn "safe to exit at layer 20 but not layer 4" because that distinction doesn't exist in the training data.

Result: the model learns to classify positions (fragile vs safe) but cannot learn *when* to exit within a position.

---

## The V2 Paradigm Shift: The Hybrid Target

To achieve earlier exits without abandoning the core "Regret-Aware" hypothesis, V2 decouples the problem into two distinct signals:
1. **Has the representation converged?** (Layer-dependent, cheap to compute)
2. **Is this token catastrophic if wrong?** (Position-dependent, expensive to compute)

The new training target for Layer $L$ at Position $P$ is a hybrid continuous value:

$$\text{Target}_{L,P} = \text{Convergence\_Score}_{L,P} \times (1.0 - \text{Task\_Regret}_P)$$

* If regret is high (1.0), the target is 0.0 (**NEVER EXIT**).
* If regret is low (0.0) but the layer hasn't converged (0.2), the target is 0.2 (**DO NOT EXIT YET**).
* If regret is low (0.0) and the layer has converged (0.99), the target is 0.99 (**SAFE TO EXIT**).

---

## Strategy 1: Layer-Dependent Targets via Logit Probing & Convergence Metrics

### Concept
Instead of solely measuring "does perturbing this token break the code?" (position-level), Phase 2a will additionally measure "has the model settled on its final prediction?" (layer-level).

### Method A: Logit Probing
At each target layer L, apply the **unembedding head** (the final `lm_head` linear layer + final layer norm) to the hidden state at layer L to get "intermediate logits."

```python
# Pseudo-code for Phase 2a
for layer_idx in valid_layers:
    # 1. Intermediate Logits
    intermediate_logits = lm_head(final_layer_norm(gathered_hidden[layer_idx][0, pos, :]))
    final_logits = lm_head(final_layer_norm(gathered_hidden[final_layer][0, pos, :]))

    # 2. Agreement & KL Divergence
    agreement = int(np.argmax(intermediate_logits) == np.argmax(final_logits))
    
    p = softmax(final_logits)
    q = softmax(intermediate_logits)
    kl_div = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
    
    convergence_score = agreement * np.exp(-kl_div)
Method B: Hidden State Convergence MetricsMeasure the physical settling of the hidden state between consecutive layers.CRITICAL NOTE: Because the model operates in bfloat16, computing cosine similarity near 1.0 suffers from severe precision loss. You must explicitly cast h_curr and h_prev to float32 before computing norms and dot products.Pythonh_curr = np.float32(hidden_states[L][pos])
h_prev = np.float32(hidden_states[L_prev][pos])
h_final = np.float32(hidden_states[44][pos])

# 1. Cosine similarity between consecutive layers
cos_sim = np.dot(h_curr, h_prev) / (norm(h_curr) * norm(h_prev))

# 2. Projection onto final hidden state (is it moving toward the final representation?)
proj_curr = np.dot(h_curr, h_final) / norm(h_final)
proj_prev = np.dot(h_prev, h_final) / norm(h_final)
proj_delta = proj_curr - proj_prev
Strategy 2: Targeted Active Learning (Optimizing Phase 2b)Phase 2b (eSurge rollouts) is the 38-hour bottleneck. We will reduce this to ~4-5 hours without losing the task regret signal by using an active learning filter.Not every token needs a rollout. A whitespace token or a closing bracket ) has near-zero variance in regret.ImplementationTargeted Sampling: Only queue Phase 2b rollouts for positions that exhibit:High entropy / low margin in the final layer (the model is actually uncertain).Non-syntax token types (variables, operators, algorithm logic).Defaulting: Assign a default regret of 0.0 to highly predictable boilerplate/syntax tokens, completely bypassing the eSurge generation queue.Strategy 3: Multi-Scale Exit Architecture (Layer-Progressive Estimator)ConceptReplace the single MLP (which sees one layer at a time in isolation) with an architecture that processes layers progressively and makes exit decisions using accumulated evidence. The model must see the trajectory of hidden states.XLA-Optimized ArchitectureRecurrent loops (like GRUs) are notoriously slow and difficult to optimize in JAX jit compilation. Instead, treat the target layers as a sequence and use a 1D Causal Convolution or Causal Attention.Input: Tensor of shape (batch, num_layers, hidden_dim + meta_features)Processing: XLA parallelizes Conv1D across the layer dimension instantly. The model natively attends to $h_4$ while evaluating $h_{20}$ to understand the delta.Output: (batch, num_layers, 1) continuous exit scores.Pythonclass LayerProgressiveEstimator(nnx.Module):
    def __init__(self, hidden_dim):
        self.feature_proj = nnx.Linear(hidden_dim + meta_dims, 128)
        # Causal Conv1D over the layer dimension
        self.conv1d = nnx.Conv(in_features=128, out_features=128, kernel_size=3, padding='CAUSAL')
        self.exit_head = nnx.Linear(128, 1)

    def __call__(self, layer_sequence_tensor):
        x = nnx.gelu(self.feature_proj(layer_sequence_tensor))
        x = nnx.gelu(self.conv1d(x))
        return jax.nn.sigmoid(self.exit_head(x)) # Continuous score [0, 1]
Training (Continuous Regret Bands)Shift from Binary Cross Entropy (BCE) to Mean Squared Error (MSE) or Huber loss.Train against the continuous Hybrid Target defined in Section 2.The estimator will learn a smoother decision boundary, mapping partial test failures (regret 0.5) appropriately.Strategy 4: Pre-computed Token-Type PriorsObservationBoilerplate tokens (:, def, return) are inherently safe to exit early. However, string decoding and matching inside the JAX training loop breaks jit compilation.ImplementationGenerate a static boolean mask of shape (vocab_size,) during preprocessing, where syntax/boilerplate tokens are 1 and semantic tokens are 0.Pass this array to device memory (jax.device_put).During training/inference, simply index this array with the input_ids to instantly retrieve the token-type feature without string manipulation.Strategy 5: Curriculum Training & Problem ComplexityCurriculum TrainingTrain the model in stages, progressively pushing exits earlier:Stage 1: Conservative Accuracy. Train against the hybrid target with standard MSE.Stage 2: Exit-Reward Optimization. Fine-tune with a loss that explicitly penalizes late exits when the representation had already converged safely.Problem Complexity EstimateDuring Phase 1, estimate problem difficulty using cheap heuristics (prompt length, number of test cases, keyword complexity like "regex" or "dynamic programming"). Feed this scalar as a meta-feature into the estimator. Simple problems should lower the exit threshold globally.Implementation RoadmapPhase A: Data Infrastructure (Week 1)Update regret_training.py (Phase 2a): Add float32 logit probing, compute convergence metrics (cosine sim, projection), and capture the pre-computed token-type boolean mask.Update regret_training.py (Phase 2b): Implement the Active Learning filter. Default obvious syntax tokens to 0.0 regret; queue only uncertain/semantic tokens for eSurge.Phase B: Architecture (Week 1-2)Update regret_estimator.py: Rewrite the Flax NNX model to accept sequence-like layer inputs (batch, layers, features) using a causal nnx.Conv mechanism.Loss Function: Shift from BCE to MSE/Huber. Compute the Hybrid Target during the dataset collation step.Phase C: Training & Evaluation (Week 2-3)Update evaluate_estimator.py: Modify to track continuous targets and evaluate layer-by-layer prediction trajectories instead of flat binary classifications.End-to-End Test: Measure actual compute savings vs. code quality on the held-out MBPP test set.Expected Outcomes & Risk AssessmentMetricCurrent (V1)Target (V2)DriverAvg Exit Layer38.6 / 4420-25 / 44Hybrid Targets & Convergence MetricsCompute Savings~12%~45-55%Layer-Progressive EstimatorTraining Time~42 hours~5-8 hoursTargeted Phase 2b Active LearningCode QualityPreservedPreservedRetaining the Task Regret signalRisks & MitigationsRiskImpactMitigationIntermediate logits may not match final output due to norm mismatchTraining targets are noisyApply final layer norm before the lm_head projectionXLA Conv1D/Attention implementation issuesTraining instabilityStart with a simple 1D Causal Convolution; verify shapes before moving to full AttentionConvergence metrics saturateFeatures become uninformativeUse log-scale: log(1.0 - cos_sim) and ensure float32 precision is strictly enforced