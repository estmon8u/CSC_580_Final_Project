# Plan: DreamerV1 → V2 → V3 → V3.5 Incremental Upgrade

## TL;DR

Upgrade the existing tiny_dreamer_highway DreamerV1 implementation to V2, then V3, then V3.5, one component at a time on a `Testing` branch. Each step is independently verifiable. V3.5 incorporates the best RSSM-based innovations from 15+ papers (2020–2026) published between DreamerV3 and Dreamer 4. Dreamer 4 (Sep 2025) uses transformers and is a full rewrite — excluded from this plan.

---

## Phase 0: Branch Setup

1. Create `Testing` branch from current HEAD: `git checkout -b Testing`

---

## Phase 1: V1 → V2 Upgrades (6 steps)

### Step 1.1: Categorical Stochastic Representations

**What**: Replace Gaussian stochastic states (30-dim vector) with 32 categorical distributions × 32 classes = 1024-dim one-hot (reshaped to 32×32 for the network, flattened to 1024 for concat with deterministic state).
**Files**:

- `src/tiny_dreamer_highway/rssm.py` — Replace `_build_prior()` / `_build_posterior()` Gaussian heads (mean+std → softmax logits). Replace `Normal` distribution with `OneHotCategorical`. Replace reparameterized sampling with straight-through gradients (sample - logits.detach() + logits).
- `src/tiny_dreamer_highway/encoder.py` — Update `LatentState` dataclass: remove `dist_mean`/`dist_std`, add `logits` (shape 32×32). Update `stochastic` field shape from (30,) to (1024,).
- `src/tiny_dreamer_highway/config.py` — Add `num_categoricals=32`, `num_classes=32` to ModelConfig. Compute `stochastic_dim = num_categoricals * num_classes`.
- `src/tiny_dreamer_highway/decoder.py` — Update input dimensions (deterministic_dim + 1024 instead of + 30).
- `src/tiny_dreamer_highway/actor.py`, `critic.py` — Update input dimensions similarly.
**Verify**: Model forward pass produces correct tensor shapes. Existing training loop runs without error for 10 steps.

### Step 1.2: KL Balancing

**What**: Replace single `kl_divergence(posterior, prior)` with two terms: dynamics loss (train prior toward posterior) and representation loss (train posterior toward prior), using stop-gradients.

- `L_dyn = KL[sg(posterior) || prior]` — trains prior only (α=0.8 weight)
- `L_rep = KL[sg(prior) || posterior]` — trains encoder only (1-α=0.2 weight)
**Files**:
- `src/tiny_dreamer_highway/world_model_step.py` — In `compute_world_model_losses()`: replace single `kl_loss` with `kl_dyn + kl_rep`. Use `torch.distributions.kl_divergence` on categorical distributions with detached logits for stop-gradient.
- `src/tiny_dreamer_highway/sequence_world_model_step.py` — Same split for sequence-level KL in `_kl_loss()`.
- `src/tiny_dreamer_highway/config.py` — Add `kl_balance=0.8` to TrainingConfig.
**Verify**: Both KL terms appear in loss logs. Total loss magnitude similar to before.

### Step 1.3: Free Nats → Keep as is (3.0)

**What**: V2 uses free nats = 1.0 for categoricals. Keep at 3.0 initially; will reduce to 1.0 in V3 phase. No changes needed here.

### Step 1.4: Straight-Through Actor Gradients (Continuous)

**What**: V2 introduces reinforce + straight-through gradients for actor. For continuous actions (highway-v0), V2 uses reparameterized gradients (which the current implementation already does via TanhTransform rsample). **No change needed** — current actor already uses reparameterized pathwise gradients which is appropriate for continuous action spaces.
**Note**: If we later add discrete action support, we'd implement REINFORCE here.

### Step 1.5: Update Decoder Loss

**What**: V2 uses MSE reconstruction loss instead of Gaussian NLL with fixed std. Since current `ObservationDecoder` uses Gaussian with fixed std=1.0, the Gaussian NLL is mathematically equivalent to MSE (up to a constant). **No functional change needed**, but can simplify code by replacing NLL with direct MSE.
**Files**:

- `src/tiny_dreamer_highway/decoder.py` — Optionally simplify `forward()` to return reconstructed tensor directly instead of wrapping in `Normal`.
- `src/tiny_dreamer_highway/world_model_step.py` — Replace `observation_dist.log_prob()` with `F.mse_loss()`.
**Verify**: Loss values are consistent (differ only by constant offset).

### Step 1.6: End-to-End V2 Validation

**What**: Run a short training (100 WM updates + 20 behavior updates) and verify:

- Categorical latent states produce meaningful reconstructions
- KL balancing terms both decrease
- Actor/critic gradients flow correctly
- No NaN/Inf in any loss term
**Verify**: `python -m pytest tests/` passes. Short training run completes without error.

---

## Phase 2: V2 → V3 Upgrades (9 steps)

### Step 2.1: Symlog/Symexp Transformations

**What**: Apply `symlog(x) = sign(x) * ln(|x| + 1)` to observation encoder inputs and use symlog squared error for decoder loss. This normalizes varying input scales.
**Files**:

- `src/tiny_dreamer_highway/` — Create `utils.py` with `symlog()` and `symexp()` functions.
- `src/tiny_dreamer_highway/encoder.py` — Apply `symlog()` to observation input in `ObservationEncoder.forward()`.
- `src/tiny_dreamer_highway/world_model_step.py` — Replace MSE with symlog squared error: `(symlog(target) - symlog(prediction))^2`.
**Verify**: Encoder outputs have similar magnitude. Reconstruction quality unchanged.

### Step 2.2: Unimix Categoricals

**What**: Mix 1% uniform distribution into categorical logits to prevent posterior collapse: `probs = (1 - 0.01) * softmax(logits) + 0.01 / num_classes`.
**Files**:

- `src/tiny_dreamer_highway/rssm.py` — In prior and posterior heads, after computing logits, apply unimix before creating distribution.
- `src/tiny_dreamer_highway/config.py` — Add `unimix_ratio=0.01` to ModelConfig.
**Verify**: Minimum probability per class ≥ 0.01/32 ≈ 0.0003. No collapsed categoricals.

### Step 2.3: KL Free Bits (reduce to 1.0)

**What**: Change free_nats from 3.0 to 1.0 and apply per-distribution (not per-batch): `max(1.0, KL[...])` applied elementwise.
**Files**:

- `src/tiny_dreamer_highway/config.py` — Change `free_nats=3.0` → `free_nats=1.0`.
- `src/tiny_dreamer_highway/world_model_step.py` — Ensure `max()` is applied per-distribution not per-batch-mean.
**Verify**: KL loss never drops below 1.0 per distribution.

### Step 2.4: Split KL Weights (β_dyn=1.0, β_rep=0.1)

**What**: V3 uses asymmetric weighting: β_dyn=1.0 (heavier on prior learning), β_rep=0.1 (lighter on encoder regularization). This is different from V2's α=0.8/0.2 split.
**Files**:

- `src/tiny_dreamer_highway/config.py` — Replace `kl_balance=0.8` with `beta_dyn=1.0`, `beta_rep=0.1`.
- `src/tiny_dreamer_highway/world_model_step.py` — Update KL loss: `beta_dyn * L_dyn + beta_rep * L_rep`.
**Verify**: KL contribution to total loss is ~0.5-2.0 range.

### Step 2.5: Symexp Twohot Reward Loss

**What**: Replace Gaussian NLL reward loss with categorical distribution over exponentially spaced bins using twohot encoding. Bins B = symexp(linspace(-20, 20, 255)).
**Files**:

- `src/tiny_dreamer_highway/utils.py` — Add `twohot_encode(x, bins)` and `symexp_bins(num_bins=255)` functions.
- `src/tiny_dreamer_highway/decoder.py` — Modify `RewardPredictor` to output 255 logits instead of scalar. Add `twohot_loss()` method.
- `src/tiny_dreamer_highway/world_model_step.py` — Replace reward Gaussian NLL with twohot cross-entropy.
**Verify**: Reward predictions (decoded as bin-weighted mean) match ground truth range.

### Step 2.6: Distributional Critic + Symexp Twohot

**What**: Replace scalar Gaussian critic with categorical critic over same symexp bins. Critic outputs 255 logits; value = weighted sum of bins.
**Files**:

- `src/tiny_dreamer_highway/critic.py` — Output 255 logits. Add `value()` method returning weighted bin mean. Loss = twohot cross-entropy against TD-λ targets.
- `src/tiny_dreamer_highway/behavior_learning.py` — Update critic loss from Gaussian NLL to twohot cross-entropy. Actor uses `critic.value()` for return estimation.
**Verify**: Critic value predictions are in reasonable range. Actor loss is finite.

### Step 2.7: EMA Target Critic

**What**: Add exponential moving average copy of critic for bootstrapping in TD-λ. Slow critic η=0.98 (update rate).
**Files**:

- `src/tiny_dreamer_highway/behavior_learning.py` — Create `target_critic` as deepcopy. After each critic update, do `target_critic.params = η * target_critic.params + (1-η) * critic.params`. Use `target_critic` for computing TD-λ targets.
- `src/tiny_dreamer_highway/config.py` — Add `critic_ema_decay=0.98`.
**Verify**: Target critic values lag behind online critic. Returns are smoother.

### Step 2.8: Return Normalization (Percentile Scaling)

**What**: Normalize returns by running percentile range: `S = EMA(Percentile(R,95) - Percentile(R,5), decay=0.99)`. Scale advantage by `max(1, S)`.
**Files**:

- `src/tiny_dreamer_highway/behavior_learning.py` — Add `ReturnNormalizer` class tracking 5th/95th percentile EMAs. Apply to returns before actor gradient: `(R_λ - V(s)) / max(1, S)`.
- `src/tiny_dreamer_highway/config.py` — Add `return_norm_decay=0.99`, `return_norm_low=0.05`, `return_norm_high=0.95`.
**Verify**: Normalized advantages have unit-ish scale. Actor gradients are stable.

### Step 2.9: Network Architecture Updates

**What**: Replace ELU activations (RSSM MLPs) and ReLU (encoder/decoder CNNs) with SiLU (swish). Add LayerNorm before each activation in MLPs. Zero-initialize output weights for reward predictor and critic.
**Files**:

- `src/tiny_dreamer_highway/rssm.py` — Replace `nn.ELU` with `nn.SiLU`. Add `nn.LayerNorm` in prior/posterior MLPs.
- `src/tiny_dreamer_highway/encoder.py` — Replace `nn.ReLU` with `nn.SiLU` in CNN.
- `src/tiny_dreamer_highway/decoder.py` — Replace `nn.ReLU` with `nn.SiLU` in decoder CNN and reward MLP. Zero-init final linear weights.
- `src/tiny_dreamer_highway/critic.py` — Replace activations with `nn.SiLU`. Add LayerNorm. Zero-init final linear.
- `src/tiny_dreamer_highway/actor.py` — Replace activations with `nn.SiLU`. Add LayerNorm.
**Verify**: All activations are SiLU. LayerNorm present. `model.reward_predictor[-1].weight` is all zeros at init.

---

## Relevant Files (all under `Final Project/CSC_580_Final_Project/src/tiny_dreamer_highway/`)

- `config.py` — Add new hyperparameters (categoricals, KL weights, unimix, EMA, return norm, bins, ensemble, Lagrangian, CPC, S5, KAN)
- `rssm.py` — Categorical stochastic state, straight-through, unimix, SiLU+LayerNorm, ensemble prior heads, S5 backbone
- `encoder.py` — LatentState dataclass update, symlog input, SiLU, BatchNorm (V3.5)
- `decoder.py` — Symlog squared error, twohot reward, SiLU, zero-init, residual frame prediction (V3.5), FastKAN predictors (V3.5)
- `world_model_step.py` — KL balancing, free bits, symlog error, twohot reward loss, ensemble loss, contrastive CPC, value-prediction loss, Barlow Twins
- `sequence_world_model_step.py` — Same KL changes for sequence training, contrastive CPC
- `actor.py` — SiLU+LayerNorm (architecture only; pathwise gradients kept)
- `critic.py` — Distributional twohot critic, EMA target, SiLU+LayerNorm, zero-init, cost critic (V3.5)
- `behavior_learning.py` — EMA target critic, return normalization, twohot critic loss, Lagrangian cost optimization (V3.5)
- `pipeline.py` — Intrinsic exploration reward injection (V3.5), cost signal propagation
- `envs/highway_factory.py` — Cost signal exposure (V3.5)
- `data/replay_buffer.py` — Store costs alongside rewards (V3.5)
- NEW `utils.py` — symlog, symexp, twohot_encode, symexp_bins helpers
- NEW `models/s5_layer.py` — S5 structured state space cell (V3.5)
- NEW `models/fastkan.py` — FastKAN layer (V3.5)

## Verification (per step)

1. Unit tests: `python -m pytest tests/` — all existing tests must pass (update expected shapes)
2. Smoke train: 50–100 WM updates + 20 behavior updates, check no NaN/Inf
3. Reconstruction quality: Visual check via prediction notebooks
4. Loss curves: All loss terms finite and trending down

## Decisions

- **Dreamer 4 excluded**: Uses transformers (not RSSM), shortcut forcing objective, designed for Minecraft-scale. Fundamentally different architecture — not an incremental upgrade.
- **Upgrade order**: V2 first (categorical representations + KL balancing are the foundation), then V3 (robustness improvements on top), then V3.5 (research-driven enhancements targeting driving performance and safety).
- **Continuous actions**: Keep reparameterized pathwise gradients (standard for continuous control). REINFORCE is only needed for discrete actions.
- **Existing features kept**: Latent overshooting (sequence_world_model_step), continue predictor, entropy regularization.
- **V3.5 philosophy**: Every V3.5 upgrade is backed by a published paper (ICML/ICLR/NeurIPS). Each step is optional and togglable via config — the agent should still work with any subset of V3.5 features enabled.

## Further Considerations

1. **Training hyperparameters**: V3 uses γ=0.997, horizon=16 (current: γ=0.99, horizon=5). Adjust after architecture is stable? Recommend tuning separately.
2. **Encoder/decoder architecture**: V3 uses larger networks. Keep current 4-layer CNN for highway-v0 (64×64 is small). Scale if needed.
3. **Mixed-precision**: V3 uses float16 for speed. Optional optimization, not architectural.

---

## Phase 3: V3 → V3.5 Upgrades (6 steps)

DreamerV3.5 is the **last generation of RSSM-based innovation** — it pulls the best ideas from 15+ papers published between V3 (Jan 2023) and Dreamer 4 (Sep 2025). Dreamer 4 abandons the RSSM entirely for a transformer with a shortcut forcing objective, so V3.5 represents the ceiling of what can be achieved within the RSSM framework. Each step below is independently verifiable and ordered by implementation priority (impact-to-effort ratio).

### Research Sources

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Plan2Explore (Sekar, Rybkin, Hafner, Pathak et al.) | 2020 | ICML | Ensemble disagreement for self-supervised exploration |
| DreamerPro (Deng, Jang, Ahn) | 2021 | — | Prototypical representations: reconstruction-free world model |
| TransDreamer (Chen, Wu, Yoon, Ahn) | 2022 | NeurIPS WS | Transformer State-Space Model replacing GRU |
| S4WM (Deng, Park, Ahn) | 2023 | NeurIPS | S4/S5 as world model backbones — outperform transformers on long-term memory |
| SafeDreamer (Huang, Ji, Xia, Zhang, Yang) | 2024 | ICLR | Lagrangian cost critic for safe RL in Dreamer |
| MuDreamer (Burchi, Timofte) | 2024 | — | Value-prediction + action-prediction as reconstruction-free representation loss |
| HRSSM (Sun, Zang, Li, Islam) | 2024 | ICML | Spatio-temporal masking + bisimulation for robust latent representations |
| DIAMOND (Alonso, Jelley, Micheli et al.) | 2024 | NeurIPS Spotlight | Diffusion world model — visual details matter |
| TD-MPC2 (Hansen, Su, Wang) | 2024 | ICLR | Scalable decoder-free world model for continuous control |
| TWISTER (Burchi, Timofte) | 2025 | — | Contrastive Predictive Coding for transformer world models — 162% HNS on Atari 100k |
| InDRiVE (Khanzada, Kwon) | 2025 | IROS sub. | Ensemble disagreement exploration for autonomous driving with Dreamer |
| KAN-Dreamer (Shi, Luan) | 2025 | — | Kolmogorov-Arnold Networks as drop-in MLP replacements in DreamerV3 |
| Dreamer-CDP (Hauri, Zenke) | 2026 | — | JEPA-style continuous deterministic prediction — reconstruction-free |
| R2-Dreamer (Morihira, Nahar et al.) | 2026 | ICLR | Barlow Twins redundancy reduction — decoder-free, no augmentation, 1.59× faster |
| Dreamer 4 (Hafner, Yan, Lillicrap) | 2025 | — | Transformer world model with shortcut forcing — excluded (full rewrite) |

---

### Step 3.1: Ensemble Disagreement Exploration

**What**: Add an ensemble of K=5 lightweight prior-prediction heads to the RSSM. The variance across their next-state predictions serves as an intrinsic curiosity reward that drives the agent toward high-uncertainty regions of the environment. During warm-start and early training the intrinsic bonus dominates; it anneals to zero as the world model becomes accurate.

**Why**: V3 explores with random actions during warm-start, then relies on a small entropy bonus in the actor loss. In sparse-reward or large environments this is insufficient — the agent may loop through the same few states and never discover rewarding transitions. Plan2Explore (ICML 2020, co-authored by Hafner) showed that ensemble disagreement almost matches oracle exploration. InDRiVE (2025) proved it works specifically for autonomous driving, achieving higher success rates and fewer infractions than DreamerV3 with fewer training steps.

**How it works**:
1. Train K=5 independent MLP heads that each predict the next stochastic state from the current deterministic state + action.
2. At each step, compute intrinsic reward = $\beta \cdot \frac{1}{K} \sum_{k=1}^{K} \lVert \hat{z}_{t+1}^{(k)} - \bar{z}_{t+1} \rVert^2$ where $\bar{z}$ is the ensemble mean.
3. Total reward = extrinsic + intrinsic. $\beta$ starts at 1.0 and anneals to 0.0 over the first 500 cycles.

**Files**:
- `src/tiny_dreamer_highway/models/rssm.py` — Add `EnsemblePriorHeads` as a new `nn.Module` containing K copies of the prior MLP. Add a `compute_disagreement(state, action)` method that returns the variance-based intrinsic reward.
- `src/tiny_dreamer_highway/training/world_model_step.py` — After computing the standard world model loss, also compute ensemble prediction loss: each head predicts the posterior stochastic state from the prior input, trained with MSE. Add `ensemble_loss` to the loss dict.
- `src/tiny_dreamer_highway/training/pipeline.py` — After collecting rewards from the environment, add the intrinsic bonus to the stored rewards in the replay buffer.
- `src/tiny_dreamer_highway/config.py` — Add to `TrainingConfig`: `exploration_ensemble_size: int = 5`, `exploration_beta_start: float = 1.0`, `exploration_beta_end: float = 0.0`, `exploration_anneal_cycles: int = 500`.

**Verify**: Loss dict includes `ensemble_loss`. Intrinsic reward is high early in training and anneals to zero. Agent explores more diverse states during warm-start compared to random actions.

---

### Step 3.2: Safety-Constrained Imagination (Lagrangian Cost Critic)

**What**: Add a second critic head (the *cost critic*) that predicts cumulative collision count from any imagined state. The actor's objective is augmented with a Lagrangian term that automatically penalizes unsafe imagined trajectories. A learned Lagrange multiplier λ is updated to enforce a constraint like "average crash rate ≤ 0.1 per episode".

**Why**: V3 has no concept of hard safety constraints. In highway driving, the reward function includes a collision penalty, but the agent can learn to tolerate occasional crashes if the speed reward outweighs the crash cost. SafeDreamer (ICLR 2024) showed that Lagrangian-based constraints inside imagination achieve nearly zero-cost (zero-crash) performance while maintaining comparable reward. For a driving domain this is directly relevant — we want a policy that is both fast and safe.

**How it works**:
1. The environment provides a binary *cost* signal: `c_t = 1.0` if a collision occurred, `0.0` otherwise.
2. The cost critic (same architecture as the reward critic) predicts expected cumulative cost from each imagined state.
3. The actor maximizes reward minus λ × predicted cost:
   - Actor loss = $-\mathbb{E}[\text{returns}] + \lambda \cdot \mathbb{E}[\text{cost\_returns}]$
4. λ is a learnable scalar, updated to satisfy the constraint:
   - $\lambda \leftarrow \max(0, \lambda + \alpha_\lambda \cdot (\mathbb{E}[\text{cost}] - d))$ where $d$ is the cost threshold.

**Files**:
- `src/tiny_dreamer_highway/models/critic.py` — Add `CostCritic` class (identical architecture to `Critic`) that predicts expected cumulative collisions.
- `src/tiny_dreamer_highway/training/behavior_learning.py` — Modify `imagine_trajectory` to also compute cost predictions from the cost critic at each imagined step. Add `train_behavior_step` changes: compute cost TD-λ returns, compute Lagrangian actor loss, update λ with dual gradient ascent.
- `src/tiny_dreamer_highway/envs/highway_factory.py` — Expose `cost` signal in the info dict: `cost = 1.0 if crashed else 0.0`.
- `src/tiny_dreamer_highway/data/replay_buffer.py` — Store `costs` alongside `rewards` in transitions.
- `src/tiny_dreamer_highway/training/pipeline.py` — Pass cost signal through the training loop.
- `src/tiny_dreamer_highway/config.py` — Add to `TrainingConfig`: `use_cost_critic: bool = True`, `cost_threshold: float = 0.1`, `lagrange_lr: float = 1e-3`, `cost_critic_lr: float = 8e-5`.

**Verify**: Cost critic loss appears in logs. λ increases when crash rate exceeds threshold, decreases when below. Agent learns policies with fewer collisions than V3 baseline.

---

### Step 3.3: Reconstruction-Free Representation Learning

**What**: Replace the pixel reconstruction loss with two lightweight auxiliary objectives: (1) a *value-prediction head* that predicts the current-step return from the latent state, and (2) a *Barlow Twins* cross-correlation loss between batches of latent features to prevent representation collapse. The pixel decoder remains in the codebase as an optional diagnostic tool (toggled via config) but is no longer part of the training gradient path.

**Why**: The pixel decoder wastes capacity on task-irrelevant detail (road textures, background, NPC colors). MuDreamer (2024) showed value-prediction is a strong representation signal; R2-Dreamer (ICLR 2026) showed Barlow Twins prevents collapse without augmentation while training 1.59× faster than V3. Dreamer-CDP (2026) confirmed JEPA-style prediction matches Dreamer's performance without reconstruction. The combined approach provides task-focused, collapse-resistant representations with a significant speedup.

**How it works**:
1. **Value-prediction head**: A small MLP takes the latent feature `[h_t; s_t]` and predicts the 1-step observed reward + discounted bootstrap value. Loss = MSE against the actual reward + V(next_state). This forces the representation to encode only what matters for predicting returns.
2. **Barlow Twins redundancy reduction**: Compute the cross-correlation matrix $C$ of the latent features `[h_t; s_t]` **across the batch dimension** (each row of C is one feature dimension, each column another). Normalize features to zero mean and unit variance per dimension first (batch normalization). Then:
   - **On-diagonal** terms ($C_{ii}$) should equal 1.0 — each feature dimension should have unit variance (not collapsed).
   - **Off-diagonal** terms ($C_{ij}$, $i \neq j$) should equal 0.0 — different feature dimensions should be uncorrelated (no redundancy).
   - Loss = $\sum_i (C_{ii} - 1)^2 + \lambda_\text{bt} \sum_{i \neq j} C_{ij}^2$
   - No augmented views or temporal pairs are needed — the redundancy reduction operates on the single batch of latent features. The Value Prediction head and RSSM already handle temporal dynamics; Barlow Twins purely prevents feature collapse and decorrelates dimensions.
3. **Batch normalization**: Add a `nn.BatchNorm1d` layer after the encoder output, before the latent features are passed to the RSSM. This prevents representation collapse (per MuDreamer's finding that BN is critical for decoder-free methods) and also provides the zero-mean/unit-variance features that Barlow Twins expects.

**Files**:
- `src/tiny_dreamer_highway/models/encoder.py` — Add `nn.BatchNorm1d(embedding_dim)` after the CNN stack output.
- `src/tiny_dreamer_highway/models/decoder.py` — No changes (decoder stays for diagnostic use).
- `src/tiny_dreamer_highway/models/world_model.py` — Add a `ValuePredictionHead` (2-layer MLP → scalar) alongside existing heads.
- `src/tiny_dreamer_highway/training/world_model_step.py` — Replace `reconstruction_loss = F.mse_loss(...)` with two new losses:
  - `value_pred_loss`: MSE between predicted return and target (reward + γ × V(next_state)).
  - `barlow_twins_loss`: Compute the batch cross-correlation matrix of latent features `[h_t; s_t]` (shape `(feature_dim, feature_dim)`), penalize diagonal ≠ 1 and off-diagonal ≠ 0.
  - When `use_reconstruction_loss=True` (config), the standard MSE is kept alongside the new losses (useful for prediction visualization); when `False`, the decoder is not trained.
- `src/tiny_dreamer_highway/config.py` — Add to `TrainingConfig`: `use_reconstruction_loss: bool = False`, `value_prediction_weight: float = 1.0`, `barlow_twins_weight: float = 1.0`, `barlow_twins_lambda: float = 5e-3`.

**Verify**: With `use_reconstruction_loss=False`, training runs ~40% faster (no decoder backward pass). Latent features do not collapse (verify via feature rank or variance across dimensions — all diagonal elements of C should be near 1.0, off-diagonal near 0.0). Reward signal still trends upward.

---

### Step 3.4: Multi-Horizon Contrastive Predictive Coding

**What**: Add a contrastive loss that trains the RSSM to predict latent states multiple steps into the future, not just one step. Using an InfoNCE objective at horizons {2, 4, 8, 16}, the model learns to maintain long-horizon consistency in latent space without the cost of multi-step pixel reconstruction.

**Why**: The standard RSSM is trained with single-step prediction only: given `(h_t, s_t, a_t)`, predict `s_{t+1}`. During imagination, errors accumulate over multi-step rollouts that the model was never explicitly trained to handle. TWISTER (Burchi & Timofte, 2025) showed that adding Contrastive Predictive Coding (CPC) to the world model achieves 162% human-normalized mean on Atari 100k — SOTA for methods without lookahead search. The contrastive loss is lightweight (no decoder needed at prediction horizons) and directly improves the quality of imagined rollouts, which is exactly what the actor depends on.

**How it works**:
1. During training, after computing the standard 1-step posterior sequence, use the prior transition model to roll forward from each time step t for k ∈ {2, 4, 8, 16} steps using the recorded actions.
2. At each horizon k, compute the predicted latent features $\hat{f}_{t+k}$ and the actual encoded features $f_{t+k}$.
3. Minimize InfoNCE contrastive loss: the predicted feature at (batch_i, step_t, horizon_k) should be most similar to the actual feature at (batch_i, step_t+k), and dissimilar to features from other batch elements (negatives).
4. Use a learned bilinear projection W: similarity = $\hat{f}^T W f$.

**Files**:
- `src/tiny_dreamer_highway/training/world_model_step.py` — Add `compute_contrastive_loss()` function. In `compute_world_model_losses()`, after computing the standard sequence of posteriors, roll the prior forward at multiple horizons and compute InfoNCE loss against the actual posteriors. Add `contrastive_loss` to the loss dict.
- `src/tiny_dreamer_highway/models/world_model.py` — Add a learnable bilinear projection `nn.Bilinear(feature_dim, feature_dim, 1)` or a small projection MLP for CPC similarity computation.
- `src/tiny_dreamer_highway/config.py` — Add to `TrainingConfig`: `cpc_horizons: list[int] = [2, 4, 8, 16]`, `cpc_weight: float = 0.5`, `cpc_temperature: float = 0.07`.
- `src/tiny_dreamer_highway/training/sequence_world_model_step.py` — Same contrastive loss for the sequence training pathway.

**Verify**: `contrastive_loss` appears in logs and decreases over training. N-step prediction accuracy (PSNR/SSIM) improves at steps 10-15 compared to V3 without CPC. Latent rollout consistency (`latent_mse`) drifts slower.

---

### Step 3.5: S5 Dynamics Backbone (Replace GRU)

**What**: Replace the GRU cell inside the RSSM with an S5 (Simplified Structured State Space) layer. The RSSM structure is preserved — prior/posterior, categorical stochastic state, KL balancing — but the recurrence mechanism becomes an S5 layer instead of a GRU.

**Why**: The GRU has limited memory capacity. S4WM (Deng, Park & Ahn, NeurIPS 2023) benchmarked RNN, Transformer, and S4/S5 as world model backbones and found that S4/S5 **outperforms transformers on long-term memory** tasks while being more efficient during both training (O(L log L) via parallel scan) and imagination. For highway-v0, episode lengths are moderate (40-400 steps), but when combined with CPC from Step 3.4, the S5 backbone provides a better foundation for multi-horizon prediction.

**How it works**:
1. Replace `self.gru = nn.GRUCell(hidden_dim, deterministic_dim)` with an S5 layer that maps `(hidden_dim,) → (deterministic_dim,)`.
2. During training (sequence mode), the S5 processes the entire time dimension in parallel via the parallel scan algorithm.
3. During imagination (step mode), the S5 runs one step at a time using its recurrent form, just like the GRU does now.
4. The S5 layer internally uses a diagonal state-space parameterization with learnable matrices (A, B, C, D).

**Files**:
- NEW `src/tiny_dreamer_highway/models/s5_layer.py` — Implement the S5 cell:
  - `S5Cell.__init__`: Initialize diagonal A matrix (complex-valued), B, C, D matrices. A is initialized with HiPPO-LegS initialization for optimal long-range memory.
  - `S5Cell.forward_recurrent(x_t, h_t)`: Single-step recurrent mode for imagination.
  - `S5Cell.forward_parallel(x_seq)`: Parallel scan over a full sequence for training.
  - Use `torch.complex64` for internal state, real-valued output.
- `src/tiny_dreamer_highway/models/rssm.py` — Replace `self.gru = nn.GRUCell(...)` with `self.s5 = S5Cell(hidden_dim, deterministic_dim, state_dim=64)`. Update `_next_deterministic` to call `self.s5.forward_recurrent()`. Add a `forward_sequence()` method that calls `self.s5.forward_parallel()` for efficient sequence training.
- `src/tiny_dreamer_highway/training/sequence_world_model_step.py` — Use the parallel scan path when processing full sequences (training), fall back to step-by-step for imagine_step (behavior learning).
- `src/tiny_dreamer_highway/config.py` — Add to `ModelConfig`: `dynamics_backbone: Literal["gru", "s5"] = "s5"`, `s5_state_dim: int = 64`.

**Verify**: Prior/posterior distributions produce correct shapes with S5 backbone. Forward pass speed is comparable or faster than GRU for sequences ≥16 steps. Imagination rollouts produce coherent trajectories. All existing tests pass (may need shape updates).

---

### Step 3.6: Minor Architecture Improvements

**What**: Two small improvements inspired by KAN-Dreamer and ResWM.

#### 3.6a: FastKAN Reward/Continue Predictors

Replace the MLP layers in `RewardPredictor` and `ContinuePredictor` with FastKAN (Radial Basis Function-based Kolmogorov-Arnold Network) layers. KAN-Dreamer (Shi & Luan, 2025) showed this is a drop-in replacement that matches MLP performance with better parameter efficiency and interpretable internal representations.

**Files**:
- NEW `src/tiny_dreamer_highway/models/fastkan.py` — Implement `FastKANLayer`: a single layer using Gaussian RBF basis functions on a learned grid. `forward(x)`: compute RBF activations then linearly combine.
- `src/tiny_dreamer_highway/models/decoder.py` — Replace `nn.Linear` + `nn.ELU` blocks in `RewardPredictor` and `ContinuePredictor` with `FastKANLayer`. Keep the same hidden dims.
- `src/tiny_dreamer_highway/config.py` — Add to `ModelConfig`: `use_kan_predictors: bool = False`, `kan_grid_size: int = 8`.

#### 3.6b: Residual Frame Prediction

Have the observation decoder predict a *residual delta* from the previous frame rather than the full frame. Since highway scenes change slowly between steps (camera is ego-centered, road scrolls smoothly), the network only needs to model the difference.

**Files**:
- `src/tiny_dreamer_highway/models/decoder.py` — In `ObservationDecoder.forward()`, accept an optional `previous_observation` tensor. When provided, the decoder output is `previous_observation + decoder_output` (decoder learns the delta).
- `src/tiny_dreamer_highway/training/world_model_step.py` — Pass the previous observation frame to the decoder for residual computation.
- `src/tiny_dreamer_highway/config.py` — Add to `ModelConfig`: `residual_decoder: bool = False`.

**Verify**: FastKAN predictors produce reward/continue predictions in the same range as MLP. Residual decoder produces non-zero deltas that, added to the previous frame, reconstruct the target frame.

---

## V3.5 Summary Table

| Component | V3 (Phase 2) | V3.5 (Phase 3) | Source Paper |
|-----------|-------------|-----------------|--------------|
| **Exploration** | Random warm-start + entropy bonus | **Ensemble disagreement** intrinsic reward (K=5 prior heads) | Plan2Explore (ICML 2020), InDRiVE (2025) |
| **Safety** | None (soft collision penalty in reward) | **Lagrangian cost critic** with auto-tuned λ | SafeDreamer (ICLR 2024) |
| **Representation loss** | Pixel reconstruction (symlog MSE) | **Value prediction + Barlow Twins** (decoder optional) | MuDreamer (2024), R2-Dreamer (ICLR 2026) |
| **Multi-horizon training** | Single-step prediction only | **Contrastive Predictive Coding** at horizons {2,4,8,16} | TWISTER (2025) |
| **Dynamics backbone** | GRU (RSSM) | **S5 layer** (RSSM structure preserved) | S4WM (NeurIPS 2023) |
| **Reward/continue heads** | MLP (SiLU+LayerNorm) | **FastKAN** (RBF-based, optional) | KAN-Dreamer (2025) |
| **Decoder** | Full frame reconstruction | **Residual delta** prediction (optional) | ResWM (2026) |
| Stochastic state | 32×32 categoricals + unimix | Same (no change) | — |
| KL balancing | β_dyn=1.0, β_rep=0.1 | Same (no change) | — |
| Value heads | Distributional twohot + EMA target | Same + **cost critic** (same arch) | SafeDreamer |
| Return normalization | Percentile scaling | Same (no change) | — |

## V3.5 Implementation Priority

Ordered by impact-to-effort ratio for the highway-v0 domain:

1. **Step 3.1 — Ensemble exploration** (easiest, directly relevant to driving, proven by InDRiVE)
2. **Step 3.2 — Cost critic** (straightforward second critic head, high value for driving safety)
3. **Step 3.3 — Reconstruction-free** (biggest speedup, moderate implementation effort)
4. **Step 3.4 — Contrastive CPC** (highest world model quality gain, moderate effort)
5. **Step 3.5 — S5 backbone** (highest effort, biggest payoff for long-horizon tasks)
6. **Step 3.6 — FastKAN + residual** (low priority, minor gains, good for experimentation)

## V3.5 New Config Fields Summary

```yaml
# TrainingConfig additions
exploration_ensemble_size: 5
exploration_beta_start: 1.0
exploration_beta_end: 0.0
exploration_anneal_cycles: 500
use_cost_critic: true
cost_threshold: 0.1
lagrange_lr: 0.001
cost_critic_lr: 0.00008
use_reconstruction_loss: false
value_prediction_weight: 1.0
barlow_twins_weight: 1.0
barlow_twins_lambda: 0.005
cpc_horizons: [2, 4, 8, 16]
cpc_weight: 0.5
cpc_temperature: 0.07

# ModelConfig additions
dynamics_backbone: "s5"       # "gru" or "s5"
s5_state_dim: 64
use_kan_predictors: false
kan_grid_size: 8
residual_decoder: false
```

## V3.5 New Files

- `src/tiny_dreamer_highway/models/s5_layer.py` — S5 structured state space cell (parallel scan + recurrent modes)
- `src/tiny_dreamer_highway/models/fastkan.py` — FastKAN layer (RBF-based Kolmogorov-Arnold Network)
