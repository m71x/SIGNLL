Regret-Aware Early Exit for Large Language Models via Counterfactual Difficulty Estimation & Layer-Progressive Convergence1. MotivationEarly-exit mechanisms for large language models (LLMs) aim to reduce inference cost by terminating computation at intermediate layers when the model is "confident enough." Existing approaches largely rely on local confidence signals such as entropy, margin, or similarity between intermediate and final-layer distributions. While effective for classification-style tasks, these methods consistently degrade performance on hierarchical reasoning tasks such as mathematics, coding, and multi-step logical inference.We argue that this degradation arises from a fundamental mismatch between confidence and risk:Not all tokens are equally important to the final task outcome.Errors early in a reasoning trajectory can compound and invalidate all downstream computation.Confidence-based early exit treats all mistakes symmetrically, ignoring downstream consequences.Initial iterations of regret-aware systems solved this by predicting position-level fragility, but encountered a spatial vs. temporal resolution problem: they learned where the code was fragile, but had no layer-dependent signal to learn when the hidden states had successfully crystallized. As a result, they exited far too late (e.g., layer 38/44), saving minimal compute.This project proposes a hybrid Regret-Convergence early-exit framework that explicitly models both the expected downstream cost of exiting early and the layer-by-layer convergence trajectory. The key idea is to decouple:Has the representation converged? (Temporal / Layer-dependent)How costly would it be to be wrong? (Spatial / Position-dependent)2. Core IdeaWe reframe early exit as an optimal stopping problem under risk.At each decoding step and layer, the model must decide whether to:Continue computation (incurring additional cost), orExit early (risking downstream task failure)The correct decision depends on a Hybrid Target combining convergence and regret. Early exit should be discouraged not when the model is uncertain, but when the consequences of a plausible mistake are large and the layer representation has not yet stabilized.To operationalize this, we introduce an XLA-Optimized Layer-Progressive Estimator trained using counterfactual perturbations that simulate realistic near-miss errors, combined with intermediate logit probing and convergence metrics.3. High-Level ArchitectureThe proposed system consists of three components:Base LLM — A pretrained autoregressive LLM (decoder-only or encoder-decoder), augmented with multiple candidate exit points at intermediate layers.Pre-computed Token-Type Priors — A static boolean mask of shape (vocab_size,) passed to device memory that flags inherently safe boilerplate/syntax tokens (e.g., :, def, return) without breaking JAX jit compilation.Layer-Progressive Estimator — A learned, causal 1D Convolution or Attention module that processes the trajectory of hidden states across layers to predict a continuous "Safe to Exit" score.At inference time, the early-exit decision is made by a halting policy that exits only when the predicted safe-to-exit score surpasses a compute-saving threshold.4. Formalizing Early Exit as the Hybrid TargetLet:$y^*$ be the output produced by full-depth inference.$R(y)$ be a task-specific reward.$y^{(\ell,t)}$ be the output produced if the model exits at layer $\ell$ at token position $t$.We define Task Regret at position $t$ as:$$\mathcal{R}_t = \mathbb{E}[\max(0, R(y^*) - R(y^{(\text{perturbed},t)}))]$$We define Convergence at layer $\ell$ and position $t$ as a measure of physical hidden-state settling and logit agreement.The Hybrid Training Target (Continuous "Safe to Exit" score) is:$$\text{Target}_{\ell,t} = \text{Convergence\_Score}_{\ell,t} \times (1.0 - \mathcal{R}_t)$$If regret is high (1.0), the target is 0.0 (NEVER EXIT).If regret is low (0.0) but the layer hasn't converged (0.2), the target is 0.2 (DO NOT EXIT YET).If regret is low (0.0) and the layer has converged (0.99), the target is 0.99 (SAFE TO EXIT).5. Why Confidence Alone Is InsufficientConfidence-based signals fail for three reasons:Token importance is highly non-uniform — An incorrect adjective rarely matters; an incorrect operator or variable often invalidates the entire solution.Hierarchical error compounding — Early reasoning errors propagate and amplify.Distributional matching is misaligned with task success — Matching the final-layer token distribution does not guarantee preservation of task-level correctness.Our framework solves this by anchoring distributional matching (convergence) to task-level consequences (regret).6. Training the Layer-Progressive EstimatorThe estimator is trained offline, using a mix of pure forward-pass metrics and active-learning perturbation rollouts.6.1 Measuring Layer Convergence (Phase 2a)Instead of solely measuring "does perturbing this token break the code?", Phase 2a additionally measures "has the model settled on its final prediction?"Method A: Logit ProbingApply the unembedding head to intermediate layers.Python# 1. Intermediate Logits
intermediate_logits = lm_head(final_layer_norm(hidden_states[L][pos]))
final_logits = lm_head(final_layer_norm(hidden_states[44][pos]))

# 2. Agreement & KL Divergence
agreement = int(np.argmax(intermediate_logits) == np.argmax(final_logits))
p = softmax(final_logits)
q = softmax(intermediate_logits)
kl_div = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
convergence_score = agreement * np.exp(-kl_div)
Method B: Hidden State Settling MetricsMeasure physical representation drift. (Critical: Cast to float32 to avoid bfloat16 precision loss when cosine similarity is ~0.999).Pythonh_curr = np.float32(hidden_states[L][pos])
h_prev = np.float32(hidden_states[L_prev][pos])
h_final = np.float32(hidden_states[44][pos])

cos_sim = np.dot(h_curr, h_prev) / (norm(h_curr) * norm(h_prev))
proj_delta = (np.dot(h_curr, h_final) - np.dot(h_prev, h_final)) / norm(h_final)
6.2 Counterfactual Perturbation (Phase 2b)To measure regret, we perturb logits at token position $t$ toward top-$k$ competing tokens, simulating a realistic near-miss error.Targeted Active Learning: Running full rollouts for every token is computationally prohibitive and unnecessary (e.g., whitespace tokens have near-zero variance in regret). We filter rollouts:Queue for Rollout: Only positions with high entropy in the final layer AND non-syntax token types (variables, operators).Default Value: Assign a default regret of 0.0 to highly predictable boilerplate/syntax tokens, bypassing the generation queue entirely.6.3 AST-Aware Granular RegretUnit tests create a brittle, non-smooth reward landscape. We augment the binary pass/fail reward with Abstract Syntax Tree (AST) distance.$$\text{Regret} = \begin{cases} 0 & \text{if tests pass} \\ 0.5 + 0.5 \times \text{AST\_Distance}(C^*, C') & \text{if tests fail} \end{cases}$$This creates a continuous "regret band," teaching the estimator that minor syntactic drift is safer than catastrophic logical collapse.6.4 Model Architecture: Layer-Progressive EstimatorThe model must see the trajectory of hidden states. We use an XLA-optimized 1D Causal Convolution across the layer dimension.Pythonclass LayerProgressiveEstimator(nnx.Module):
    def __init__(self, hidden_dim):
        self.feature_proj = nnx.Linear(hidden_dim + meta_dims, 128)
        self.conv1d = nnx.Conv(in_features=128, out_features=128, kernel_size=3, padding='CAUSAL')
        self.exit_head = nnx.Linear(128, 1)

    def __call__(self, layer_sequence_tensor):
        x = nnx.gelu(self.feature_proj(layer_sequence_tensor))
        x = nnx.gelu(self.conv1d(x))
        return jax.nn.sigmoid(self.exit_head(x)) # Continuous Target [0, 1]
The loss function is Mean Squared Error (MSE) or Huber loss against the continuous Hybrid Target.7. Advanced Training MethodologiesTo combat exposure bias and hardware limitations, the training methodology includes:DAgger-Style Iterative Training: The initial dataset is off-policy (derived from full-depth trajectories). We run a second iteration generating training data using the active early-exit policy, collecting on-policy states to calibrate the estimator against the actual drift it causes.Curriculum Training: - Stage 1: Train against the hybrid target with standard MSE (Conservative Accuracy).Stage 2: Fine-tune with a loss that explicitly penalizes late exits when the representation had already converged safely (Exit-Reward Optimization).Problem Complexity Meta-Features: Estimate problem difficulty during Phase 1 using prompt length, test case count, and keyword complexity (e.g., "dynamic programming"). Feed this scalar into the estimator to lower the exit threshold globally for simple problems.8. Evaluation PlanTasks: Mathematical reasoning (GSM8K, MATH), Coding tasks (HumanEval, MBPP).Baselines: Confidence-only early exit, KL-to-final early exit, Fixed-depth truncation.Metrics:Compute vs. reward curve.Worst-case performance degradation.Average exit layer (Target: 20-25 / 44).9. 🚀 Project Execution PlanPhase 0 — Preparation + Model ChoiceSelect model (Qwen2.5-Coder-14B) and initialize JAX/EasyDeL 32-TPU mesh.Set up AST-aware unit test reward environment.Phase 1 — Build Early Exit Infrastructure & BaselinesGenerate greedy decode baseline ($y^*$) and baseline rewards ($R^*$).Extract problem complexity meta-features.Phase 2a — Forward Passes (Convergence Data)Run full forward passes on baselines.Compute $float32$ intermediate logits, KL divergence, and cosine similarities.Pre-compute and load the static token-type boolean mask into device memory.Phase 2b — Targeted Perturbation EngineApply Active Learning filter: skip boilerplate tokens (default Regret = 0.0).For uncertain/semantic tokens, apply directional logit perturbation.Execute controlled sampling + greedy rollout.Compute continuous AST-Aware Regret.Phase 3 — Train Layer-Progressive EstimatorCollate Phase 2a convergence metrics and Phase 2b regret into the Hybrid Target.Train the nnx.Conv1D model using MSE/Huber loss.Perform Iteration 1 (DAgger) on-policy trajectory sampling to fix exposure bias.Phase 4 — Joint Halting Policy & TestingDeploy the continuous "Safe to Exit" threshold.Measure actual compute savings vs. quality degradation curve.10. Frontier Extensions for Production (Phase 6)

Hardware-Aware Token Routing (Mixture-of-Depths): To realize true wall-clock speedups on TPUs, replace standard loop-halting with static routing. The estimator selects the top $K\%$ of tokens to continue, routing the remaining $(1-K)\%$ directly to the residual stream to maintain static tensor shapes for XLA compiling.

Speculative Early Exit (Draft-and-Verify): Use the early-exit mechanism to rapidly draft $N$ tokens, then feed all $N$ tokens back into the full 44-layer model in a single parallel forward pass to verify logits. This guarantees the final output perfectly matches the 14B parameter model's quality while still leveraging the massive speedup of the early-exit drafts.

---

## 11. V6 Results Analysis & The Fundamental Bottleneck

The v6 progressive Conv1D estimator achieves **98.8% validation accuracy** with MSE of 0.0045. The estimator is essentially solved — it near-perfectly predicts when a layer's hidden state has converged. But the evaluation reveals a deeper problem:

**Per-layer agreement rates (argmax match with final layer):**
| Layer | Agreement | Mean KL Target |
|-------|-----------|----------------|
| 4     | 0.1%      | 0.926          |
| 8     | 0.0%      | 0.926          |
| 12    | 0.1%      | 0.922          |
| 16    | 0.3%      | 0.916          |
| 20    | 0.5%      | 0.914          |
| 24    | 1.0%      | 0.912          |
| 28    | 2.1%      | 0.905          |
| 32    | 17.1%     | 0.797          |
| 36    | 29.7%     | 0.663          |
| 40    | 48.9%     | 0.493          |
| 44    | 83.5%     | 0.173          |

**The estimator is not the bottleneck.** The model's intermediate layers genuinely cannot produce correct outputs — the representations haven't converged. At layer 20, only 0.5% of tokens match the final layer's prediction. No amount of better estimation changes this.

At the safest threshold (0.01), early exit saves only **1.2% compute** with 99.9% safety. At threshold 0.05, 66.9% of decisions exit but the average exit layer is 42.0 — saving only 4.4% compute because nearly all exits happen at the last 1-2 layers.

**Conclusion:** To achieve meaningful compute savings (>30%), we must go beyond estimation and change the underlying compute dynamics — either making intermediate layers produce better outputs, or restructuring how computation flows through the model.

---

## 12. Next-Generation Strategies for Significant Early Exit

The following strategies target the root cause: intermediate layers lack the capacity to produce correct outputs. Each is ordered by expected impact and feasibility.

### Strategy A: Self-Speculative Decoding (Zero Quality Loss, ~40-60% Savings)

**Impact: VERY HIGH | Feasibility: HIGH | Risk: LOW**

**Core Insight:** Instead of trying to make early exit lossless (impossible for 48→20 layer jumps), use early exit as a *draft mechanism* and verify cheaply. This is self-speculative decoding — the same model serves as both drafter (shallow) and verifier (full-depth).

**Method:**
1. Run the first token through all 48 layers normally.
2. For the next $K$ tokens, exit at the estimator's recommended layer (e.g., layer 36-40).
3. After $K$ drafted tokens, run a single full-depth forward pass over all $K$ tokens in parallel (exploiting KV-cache and batch parallelism).
4. Compare drafted logits vs verified logits. Accept tokens that match; re-generate from the first mismatch.

**Why this works:**
- The verification pass processes $K$ tokens in parallel — on TPU, this takes roughly the same wall-clock time as a single token (memory-bound, not compute-bound).
- Even with a conservative draft acceptance rate of 60%, we save ~40% of sequential forward passes.
- Quality is **identical** to full-depth inference — every accepted token has been verified.
- The regret estimator makes the drafter smarter: it knows which tokens are safe to draft aggressively (exit at layer 32) vs conservatively (exit at layer 44).

**Architecture:**
```python
def self_speculative_decode(model, prompt, estimator, K=8):
    """Generate tokens using self-speculative decoding."""
    tokens = prompt
    while not done:
        # Draft K tokens using early exit
        draft_tokens = []
        draft_exit_layers = []
        for i in range(K):
            hidden_states = model.forward_to_layer(tokens, max_layer=48)
            exit_layer = estimator.recommend_exit(hidden_states)
            draft_logit = model.lm_head(hidden_states[exit_layer])
            draft_tokens.append(argmax(draft_logit))
            draft_exit_layers.append(exit_layer)
            tokens = append(tokens, draft_tokens[-1])

        # Verify: single full-depth pass over all K draft tokens
        verified_logits = model.full_forward(tokens[-K:])  # parallel over K

        # Accept longest matching prefix
        for i in range(K):
            if argmax(verified_logits[i]) != draft_tokens[i]:
                # Reject from position i onward, use verified token
                tokens = tokens[:-(K-i)] + [argmax(verified_logits[i])]
                break
```

**Expected savings:**
- With draft acceptance ~70% and K=8: effectively 1 full pass per ~5.6 tokens → ~44% compute reduction.
- With estimator-guided draft depth (layer 36 for safe tokens): additional ~18% savings on draft passes.
- **Total: ~50% compute savings with zero quality degradation.**

### Strategy B: Intermediate Layer Self-Distillation (Make Early Layers Exit-Ready)

**Impact: VERY HIGH | Feasibility: MEDIUM | Risk: MEDIUM**

**Core Insight:** The reason layer 20 has 0.5% agreement with the final layer is that it was never *trained* to produce correct logits — only the final layer receives gradient signal from the language modeling objective. Self-distillation adds auxiliary losses at intermediate layers, teaching them to approximate the final layer's output.

**Method:**
1. Freeze the pretrained model's parameters.
2. At each target exit layer $L$, add a lightweight **exit head**: a 2-layer MLP that maps $h_L$ → logits.
3. Train each exit head to minimize KL divergence with the full model's output distribution:

$$\mathcal{L}_{\text{distill}} = \sum_{L \in \text{exit\_layers}} w_L \cdot D_{KL}\Big(\text{softmax}(z_{48}/\tau) \;\|\; \text{softmax}(f_L(h_L)/\tau)\Big)$$

where $f_L$ is the exit head at layer $L$, $z_{48}$ is the full-depth logits, and $\tau$ is a temperature parameter.

4. The exit heads are tiny (~1M params each) vs the 14B parameter base model. Training them requires only forward passes through the frozen model + backprop through the exit heads.

**Why this changes the game:**
- Currently, the unembedding head (designed for layer 48's representation space) is applied to layer 20's representation, which lives in a completely different subspace. The exit head learns the *transformation* from layer 20's space to the output space.
- Papers on early exit with distillation (CALM, LayerSkip, DeeBERT) consistently show this raises intermediate layer agreement from <5% to 60-80%.
- Combined with the regret estimator, this gives us two independent signals: "can this layer produce the right answer?" (distillation) AND "would a wrong answer be catastrophic?" (regret).

**Architecture:**
```python
class ExitHead(nnx.Module):
    """Lightweight MLP to transform intermediate hidden states to logits."""
    def __init__(self, hidden_dim, vocab_size, rngs):
        self.up_proj = nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs)
        self.down_proj = nnx.Linear(hidden_dim * 2, vocab_size, rngs=rngs)
        self.ln = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, hidden_state):
        x = self.ln(hidden_state)
        return self.down_proj(nnx.gelu(self.up_proj(x)))
```

**Training cost:** ~2-4 hours on the TPU cluster (forward-only through frozen model, backprop only through small exit heads).

**Expected impact:** Layer 20 agreement jumps from 0.5% → 50-70%. Combined with regret gating, safe tokens can exit at layer 20 while fragile tokens continue to layer 48.

### Strategy C: Hidden State Extrapolation (Predict Final State from Intermediate State)

**Impact: HIGH | Feasibility: MEDIUM | Risk: MEDIUM**

**Core Insight:** Instead of asking "has layer $L$'s hidden state converged to the final layer?", train a small network to *predict* what the final layer's hidden state would be, given only the intermediate state. This is a residual prediction problem.

**Method:**
1. Collect paired data: $(h_L, h_{48})$ for each target layer $L$.
2. Train a lightweight **extrapolation network** per layer:
$$\hat{h}_{48} = h_L + g_L(h_L)$$
where $g_L$ is a small residual MLP that predicts the delta $h_{48} - h_L$.

3. At inference, apply $g_L$ to get $\hat{h}_{48}$, then apply the standard unembedding head.
4. If the extrapolated logits match the previous token's pattern (low entropy, high confidence), exit.

**Why this is different from Strategy B:**
- Strategy B trains new exit heads in a different output space per layer.
- Strategy C predicts in the *same* hidden state space as the final layer, reusing the existing unembedding head.
- The residual structure ($h_L + g_L(h_L)$) exploits the fact that transformer layers compute residual updates — $g_L$ learns to approximate the aggregate effect of layers $L+1$ through 48.

**Architecture:**
```python
class LayerExtrapolator(nnx.Module):
    """Predicts h_final from h_L via residual prediction."""
    def __init__(self, hidden_dim, rngs):
        self.net = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim // 4, rngs=rngs),
            lambda x: nnx.gelu(x),
            nnx.Linear(hidden_dim // 4, hidden_dim, rngs=rngs),
        )
    def __call__(self, h_intermediate):
        return h_intermediate + self.net(h_intermediate)  # residual
```

**Training data:** Already collected in Phase 2a — we have hidden states at all 11 target layers for 4210 positions. Train to minimize $\|g_L(h_L) - (h_{48} - h_L)\|^2$.

**Expected impact:** If the extrapolation achieves >80% top-1 agreement with the true final logits at layer 20, tokens can exit 24 layers earlier.

### Strategy D: Exit Momentum & Cross-Token Context

**Impact: MEDIUM-HIGH | Feasibility: HIGH | Risk: LOW**

**Core Insight:** Code tokens exhibit strong locality of difficulty — a sequence of simple tokens (whitespace, brackets, common keywords) is usually followed by another simple token. The current estimator decides independently per position. Adding cross-token context enables more aggressive exits by exploiting sequential patterns.

**Method 1: Exit Momentum (Simple, No Retraining)**
Track a running exponential moving average of exit layers over the last $N$ tokens:
$$\bar{L}_t = \alpha \cdot L_t + (1 - \alpha) \cdot \bar{L}_{t-1}$$

When $\bar{L}_t$ is low (recent tokens exiting early), bias the current exit threshold downward. When $\bar{L}_t$ is high (recent tokens going deep), be conservative.

```python
def exit_with_momentum(estimator_score, momentum_avg, alpha=0.1):
    # Dynamically adjust threshold based on recent exit history
    momentum_bias = (44 - momentum_avg) / 44  # 0 when deep, 1 when shallow
    adjusted_threshold = base_threshold * (1.0 + 0.5 * momentum_bias)
    return estimator_score > adjusted_threshold
```

**Method 2: Sliding Window Exit Context (Requires Retraining)**
Extend the estimator's input to include exit decisions and scores from the previous $W$ token positions. The Conv1D architecture naturally supports this by adding a second convolution dimension over the position axis:

```python
# Input shape: (batch, positions_window, layers, features + prev_exit_scores)
# Conv2D: kernel (3, 3) — causal over both position and layer dimensions
```

**Expected impact:** 10-20% additional compute savings on "easy" code regions (boilerplate, imports, simple assignments) where the momentum signal is strongest.

### Strategy E: Mixture-of-Depths with Static Token Routing

**Impact: VERY HIGH | Feasibility: LOW (requires model surgery) | Risk: HIGH**

**Core Insight:** True wall-clock speedups on TPUs require static tensor shapes — dynamic halting (where different tokens exit at different layers) creates irregular shapes that XLA cannot optimize. Mixture-of-Depths (MoD) solves this by making a binary routing decision at each layer: does this token *participate* in this layer's computation, or skip directly via the residual stream?

**Method:**
1. At each layer, the regret estimator (or a lightweight router) selects the top $K\%$ of tokens to process.
2. The remaining $(1-K)\%$ skip the layer entirely — their hidden states pass through unchanged via the residual connection.
3. Because exactly $K\%$ of tokens are selected at each layer, all tensor shapes are static and XLA can fully optimize.

**Architecture:**
```python
def mixture_of_depths_layer(hidden_states, layer_fn, router, capacity_ratio=0.5):
    """Process only top-K tokens through this layer; rest skip via residual."""
    scores = router(hidden_states)  # (batch, seq_len, 1)
    k = int(seq_len * capacity_ratio)
    top_k_indices = jnp.argsort(scores, axis=1)[:, -k:, :]

    # Gather top-K tokens, process, scatter back
    selected = jnp.take_along_axis(hidden_states, top_k_indices, axis=1)
    processed = layer_fn(selected)  # Standard transformer layer
    output = hidden_states.at[top_k_indices].set(processed)
    return output
```

**Training:** Requires fine-tuning the base model with the router in the loop (end-to-end). This is expensive (~full pretraining cost) but produces a model that natively supports variable-depth computation.

**Expected impact:** At capacity_ratio=0.5 for layers 1-32 and 1.0 for layers 33-48, theoretical compute reduction is ~33%. With regret-aware routing (fragile tokens always get full depth), quality preservation is much better than uniform MoD.

**Note:** This is a production-grade technique that requires significant engineering. Recommended only after Strategies A-D are validated.

### Strategy F: Asymmetric Exit-Aware Loss with Compute Penalty

**Impact: MEDIUM | Feasibility: HIGH | Risk: LOW**

**Core Insight:** The current MSE loss treats over-prediction and under-prediction symmetrically. But the costs are asymmetric:
- **False positive** (predict "safe to exit" when it's not): causes quality degradation. Cost scales with regret.
- **False negative** (predict "not safe" when it is): only costs compute. Cost scales with how many layers are wasted.

An asymmetric loss pushes the estimator to be aggressive about exiting at later layers (where it's usually safe) while remaining conservative at earlier layers (where mistakes are catastrophic).

**Loss function:**
$$\mathcal{L} = \sum_{i} \begin{cases}
\beta_{\text{miss}} \cdot (p_i - t_i)^2 \cdot t_i & \text{if } p_i < t_i \text{ (under-predict, false alarm)} \\
\beta_{\text{alarm}} \cdot (p_i - t_i)^2 \cdot (1 + \lambda \cdot \ell_i / L) & \text{if } p_i > t_i \text{ (over-predict, dangerous miss)}
\end{cases}$$

where $\ell_i$ is the layer index and $\lambda$ controls how much extra penalty dangerous misses at early layers receive.

**Expected impact:** Shifts the compute-safety Pareto frontier — at the same safety level, achieves 5-15% more compute savings because the estimator learns to be more aggressive where it can afford to be.

---

## 13. Recommended Implementation Order

| Priority | Strategy | Impact | Effort | Dependencies |
|----------|----------|--------|--------|--------------|
| 1 | **A: Self-Speculative Decoding** | ~50% savings, zero quality loss | 1-2 weeks | Estimator (done) |
| 2 | **D: Exit Momentum** | +10-20% savings | 2-3 days | None |
| 3 | **F: Asymmetric Loss** | +5-15% savings | 1-2 days | Retrain estimator |
| 4 | **B: Self-Distillation** | Layer 20 agreement 0.5%→60% | 2-3 weeks | TPU cluster time |
| 5 | **C: Hidden State Extrapolation** | Alternative to B | 1-2 weeks | Phase 2a data (done) |
| 6 | **E: Mixture-of-Depths** | ~33% savings, production-grade | 4-8 weeks | Model fine-tuning |

**Phase 5 Roadmap:**
- **Week 1:** Implement Strategy A (self-speculative decoding) using the existing estimator. This gives immediate, risk-free compute savings.
- **Week 1-2:** Implement Strategy D (exit momentum) as a zero-cost addon to Strategy A's draft mechanism.
- **Week 2:** Retrain estimator with Strategy F (asymmetric loss) to improve the draft acceptance rate.
- **Week 3-5:** Implement Strategy B (self-distillation) to fundamentally improve intermediate layer quality, enabling the speculative drafter to exit much earlier (layer 20-28 instead of 36-40).
- **Week 6+:** If Strategy B succeeds, evaluate whether Strategy E (MoD) is worth the engineering investment for production deployment.