# Plan: DreamerV1 → V2 → V3 Incremental Upgrade

## TL;DR

Upgrade the existing tiny_dreamer_highway DreamerV1 implementation to V2 then V3, one component at a time on a `Testing` branch. Each step is independently verifiable. Dreamer 4 (Sep 2025) uses transformers and is a full rewrite — excluded from this plan.

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

- `config.py` — Add new hyperparameters (categoricals, KL weights, unimix, EMA, return norm, bins)
- `rssm.py` — Categorical stochastic state, straight-through, unimix, SiLU+LayerNorm
- `encoder.py` — LatentState dataclass update, symlog input, SiLU
- `decoder.py` — Symlog squared error, twohot reward, SiLU, zero-init
- `world_model_step.py` — KL balancing, free bits, symlog error, twohot reward loss
- `sequence_world_model_step.py` — Same KL changes for sequence training
- `actor.py` — SiLU+LayerNorm (architecture only; pathwise gradients kept)
- `critic.py` — Distributional twohot critic, EMA target, SiLU+LayerNorm, zero-init
- `behavior_learning.py` — EMA target critic, return normalization, twohot critic loss
- NEW `utils.py` — symlog, symexp, twohot_encode, symexp_bins helpers

## Verification (per step)

1. Unit tests: `python -m pytest tests/` — all existing tests must pass (update expected shapes)
2. Smoke train: 50–100 WM updates + 20 behavior updates, check no NaN/Inf
3. Reconstruction quality: Visual check via prediction notebooks
4. Loss curves: All loss terms finite and trending down

## Decisions

- **Dreamer 4 excluded**: Uses transformers (not RSSM), shortcut forcing objective, designed for Minecraft-scale. Fundamentally different architecture — not an incremental upgrade.
- **Upgrade order**: V2 first (categorical representations + KL balancing are the foundation), then V3 (robustness improvements on top).
- **Continuous actions**: Keep reparameterized pathwise gradients (standard for continuous control). REINFORCE is only needed for discrete actions.
- **Existing features kept**: Latent overshooting (sequence_world_model_step), continue predictor, entropy regularization.

## Further Considerations

1. **Training hyperparameters**: V3 uses γ=0.997, horizon=16 (current: γ=0.99, horizon=5). Adjust after architecture is stable? Recommend tuning separately.
2. **Encoder/decoder architecture**: V3 uses larger networks. Keep current 4-layer CNN for highway-v0 (64×64 is small). Scale if needed.
3. **Mixed-precision**: V3 uses float16 for speed. Optional optimization, not architectural.
