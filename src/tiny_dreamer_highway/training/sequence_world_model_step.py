"""Sequence-level world-model training for DreamerV1.

This module implements the core training step for the world model.  Instead
of processing single transitions independently, it trains on *sequences*
sampled from the replay buffer so that the RSSM can learn temporal dynamics
across multiple time steps.

Key design decisions:

* **Vectorized loss computation** — observations, rewards, and distribution
  parameters are stacked into ``(B, T, ...)`` tensors so that reconstruction,
  reward, continue, and KL losses are computed in a single GPU kernel pass
  rather than inside a Python time-step loop.
* **Free-nats clamping** — the KL divergence is clamped from below at
  ``free_nats`` (typically 3.0 nats) to prevent the posterior from
  collapsing to the prior too early in training.  This lets the model
  first learn useful representations before the KL penalty kicks in.
* **Latent overshooting** (optional) — rolls out the prior for multiple
  steps from each posterior and penalizes divergence from the actual
  posteriors at each horizon, encouraging the learned dynamics to stay
  accurate over longer prediction windows.

Reference: Hafner et al., "Dream to Control" (ICLR 2020), Section 3.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from contextlib import nullcontext

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, optim

from tiny_dreamer_highway.models.encoder import LatentState
from tiny_dreamer_highway.models.world_model import TinyWorldModel, WorldModelOutput
from tiny_dreamer_highway.training.world_model_step import (
    _backward_and_step,
    compute_world_model_losses,
    categorical_kl_divergence,
    _raw_categorical_kl,
)
from tiny_dreamer_highway.types import ReplaySequenceBatch, Transition


def _compute_vectorized_losses(
    observations: Tensor,
    rewards: Tensor,
    reconstructions: Tensor,
    predicted_rewards: Tensor,
    predicted_continues: Tensor | None,
    post_logits: Tensor,
    prior_logits: Tensor,
    *,
    terminal_targets: Tensor | None,
    free_nats: float,
    kl_balance: float,
    continue_loss_weight: float,
    reward_std: float,
) -> dict[str, Tensor]:
    """Compute all world-model losses in one vectorized pass.

    Accepts pre-stacked ``(B, T, ...)`` tensors so that every loss term
    is evaluated with a single batch of tensor operations instead of a
    Python loop over time steps.  This yields a significant speed-up on
    GPU because it avoids per-step kernel launch overhead.

    Loss components:

    1. **Reconstruction loss** — MSE between the decoder's predicted
       image and the true observation (DreamerV2 style).
    2. **Reward loss** — negative Gaussian log-probability of the real
       reward under the reward predictor’s output.
    3. **Continue loss** — binary cross-entropy between the continue
       predictor and ``1 − terminal`` (only when a continue model exists).
    4. **KL loss** — analytic KL divergence between the posterior and
       prior Gaussian distributions, clamped at ``free_nats``.

    Args:
        observations:       Ground-truth images,        shape ``(B, T, C, H, W)``.
        rewards:            Ground-truth scalar rewards, shape ``(B, T)``.
        reconstructions:    Decoder output,              shape ``(B, T, C, H, W)``.
        predicted_rewards:  Reward head output,          shape ``(B, T, 1)``.
        predicted_continues: Continue head logits or None, shape ``(B, T, 1)``.
        post_means/stds:    Posterior Gaussian params,   shape ``(B, T, stoch_dim)``.
        prior_means/stds:   Prior Gaussian params,       shape ``(B, T, stoch_dim)``.
        terminal_targets:   Boolean terminal flags or None, shape ``(B, T)``.
        free_nats:          KL floor — KL below this value is not penalized.
        continue_loss_weight: Scaling factor for the continue loss.
        reward_std:         Fixed std of the reward Gaussian.

    Returns:
        Dict of named loss tensors (scalar, on the same device).
    """
    B, T = observations.shape[:2]
    device = observations.device

    # ── Target preparation ───────────────────────────────────────────
    # Cast observations to float (uint8 → [0,1]) and align reward shape
    # with the reward predictor's (B, T, 1) output.
    target_obs = observations.to(dtype=reconstructions.dtype)
    if observations.dtype == torch.uint8:
        target_obs = target_obs / 255.0
    target_rew = rewards.unsqueeze(-1).to(dtype=predicted_rewards.dtype)  # (B, T, 1)

    # ── Reconstruction loss (MSE — DreamerV2) ────────────────────────
    # DreamerV2 replaces the Gaussian NLL with direct MSE, which is
    # equivalent up to a constant when the decoder std is fixed.
    reconstruction_loss = F.mse_loss(reconstructions, target_obs)

    # ── Reward loss (Gaussian log-prob) ──────────────────────────────
    # Same Gaussian log-prob formulation as reconstruction, but for a
    # scalar reward with its own fixed variance.
    rew_sq_err = (predicted_rewards - target_rew).pow(2)
    rew_var = reward_std ** 2
    rew_log_prob = -0.5 * (
        math.log(2 * math.pi) + math.log(rew_var) + rew_sq_err.squeeze(-1) / rew_var
    )  # (B, T)
    reward_loss = -rew_log_prob.mean()

    # ── Continue loss (BCE) ──────────────────────────────────────────
    # Predicts whether the episode continues (1) or terminates (0).
    # Only active when the config enables a continue predictor.
    continue_loss = torch.zeros((), device=device, dtype=reconstructions.dtype)
    if terminal_targets is not None and predicted_continues is not None:
        # Invert terminals → continue targets: continue = 1 − terminated
        continue_targets = (
            1.0 - terminal_targets.unsqueeze(-1).to(dtype=predicted_continues.dtype)
        )  # (B, T, 1)
        continue_loss = F.binary_cross_entropy_with_logits(
            predicted_continues, continue_targets,
        )

    # ── KL divergence (DreamerV2 balanced categorical KL) ──────────
    # Split into dynamics loss and representation loss with stop-
    # gradients so the prior and posterior receive separate learning
    # signals (Hafner et al., 2021).
    kl_loss, kl_dyn, kl_rep = categorical_kl_divergence(
        post_logits, prior_logits,
        balance=kl_balance, free_nats=free_nats,
    )
    raw_kl = _raw_categorical_kl(post_logits, prior_logits)

    return {
        "reconstruction_loss": reconstruction_loss,
        "reconstruction_mse": reconstruction_loss.detach(),
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
        "kl_loss_raw": raw_kl.detach(),
        "kl_dynamics": kl_dyn.detach(),
        "kl_representation": kl_rep.detach(),
    }


def compute_latent_overshooting_losses(
    model: TinyWorldModel,
    outputs: list[WorldModelOutput],
    actions: Tensor,
    *,
    overshooting_horizon: int,
) -> dict[str, Tensor]:
    """Compute latent overshooting losses for multi-step prediction accuracy.

    Latent overshooting encourages the learned dynamics to remain accurate
    over multiple imagination steps, not just one-step predictions.  For
    each posterior state in the training sequence, the prior is rolled out
    for up to ``overshooting_horizon`` steps, and the resulting imagined
    distributions are compared against the actual posteriors at those
    future time steps.

    Two losses are computed per (start, offset) pair:

    * **KL divergence**: penalizes the imagined prior distribution for
      drifting from the true posterior.
    * **Feature MSE**: penalizes the imagined latent feature vector
      ``[h_t ; s_t]`` for diverging from the ground-truth feature.

    Both are averaged over all valid (start, offset) pairs.

    Args:
        model:     The world model (only its RSSM is used for rollouts).
        outputs:   Per-step WorldModelOutput from the observe pass.
        actions:   Raw actions tensor, shape ``(B, T, action_dim)``.
        overshooting_horizon: Maximum number of steps to roll forward.

    Returns:
        Dict with ``overshooting_kl_loss``, ``overshooting_feature_mse``,
        and ``overshooting_pairs`` (count of valid comparison pairs).
    """
    if overshooting_horizon <= 0 or len(outputs) <= 1:
        zero = torch.zeros((), device=actions.device)
        return {
            "overshooting_kl_loss": zero,
            "overshooting_feature_mse": zero,
            "overshooting_pairs": zero,
        }

    max_horizon = min(overshooting_horizon, len(outputs) - 1)
    kl_total = torch.zeros((), device=actions.device)
    feature_mse_total = torch.zeros((), device=actions.device)
    pair_count = 0

    for start_index in range(len(outputs) - 1):
        start_state = outputs[start_index].posterior_state
        if start_state.deterministic is None or start_state.stochastic is None:
            continue

        rollout_horizon = min(max_horizon, len(outputs) - start_index - 1)
        # Slice the actions for the rollout window starting one step ahead
        rollout_actions = actions[:, start_index + 1 : start_index + 1 + rollout_horizon]

        # Detach start state so overshooting gradients don’t flow back
        # into the posterior computation — only the prior is trained here.
        detached_start_state = LatentState(
            deterministic=start_state.deterministic.detach(),
            stochastic=start_state.stochastic.detach(),
        )
        imagined_states = model.rssm.imagine_rollout(detached_start_state, rollout_actions)

        for offset, imagined_state in enumerate(imagined_states, start=1):
            target_state = outputs[start_index + offset].posterior_state
            if (
                imagined_state.logits is None
                or target_state.logits is None
                or target_state.deterministic is None
                or target_state.stochastic is None
            ):
                continue

            # Compare imagined distribution against the actual posterior
            # (detached so the posterior is treated as a fixed target).
            kl_total = kl_total + _raw_categorical_kl(
                target_state.logits.detach(),
                imagined_state.logits,
            )

            # Also penalize the concatenated latent feature [h;s] directly
            target_features = torch.cat(
                [target_state.stochastic.detach(), target_state.deterministic.detach()],
                dim=-1,
            )
            feature_mse_total = feature_mse_total + torch.mean((imagined_state.features - target_features) ** 2)
            pair_count += 1

    if pair_count == 0:
        zero = torch.zeros((), device=actions.device)
        return {
            "overshooting_kl_loss": zero,
            "overshooting_feature_mse": zero,
            "overshooting_pairs": zero,
        }

    return {
        "overshooting_kl_loss": kl_total / pair_count,
        "overshooting_feature_mse": feature_mse_total / pair_count,
        "overshooting_pairs": torch.tensor(float(pair_count), device=actions.device),
    }


def stack_sequence_batch(sequences: list[list[Transition]]) -> ReplaySequenceBatch:
    """Assemble a batch of transition sequences into contiguous NumPy arrays.

    Each inner list is one sampled sequence of length ``T``.  The outer
    list contains ``B`` such sequences.  The function stacks them into
    ``(B, T, ...)`` arrays suitable for world-model training.

    Returns:
        A ``ReplaySequenceBatch`` named-tuple with fields:
        ``observations``, ``actions``, ``rewards``, ``next_observations``,
        ``dones``, ``terminals``, ``truncations``.
    """
    if not sequences or not sequences[0]:
        raise ValueError("sequences must be non-empty")

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    terminals = []
    truncations = []
    for sequence in sequences:
        observations.append([transition.observation for transition in sequence])
        actions.append([transition.action for transition in sequence])
        rewards.append([transition.reward for transition in sequence])
        next_observations.append([transition.next_observation for transition in sequence])
        dones.append([transition.done for transition in sequence])
        terminals.append([transition.terminated for transition in sequence])
        truncations.append([transition.truncated for transition in sequence])

    return ReplaySequenceBatch(
        observations=np.asarray(observations, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.uint8),
        dones=np.asarray(dones, dtype=np.bool_),
        terminals=np.asarray(terminals, dtype=np.bool_),
        truncations=np.asarray(truncations, dtype=np.bool_),
    )


def compute_sequence_world_model_losses(
    model: TinyWorldModel,
    observations: Tensor,
    actions: Tensor,
    rewards: Tensor,
    *,
    dones: Tensor | None = None,
    terminals: Tensor | None = None,
    kl_weight: float = 1.0,
    kl_balance: float = 0.8,
    free_nats: float = 3.0,
    continue_loss_weight: float = 1.0,
    overshooting_horizon: int = 0,
    overshooting_kl_weight: float = 0.0,
) -> tuple[list[WorldModelOutput], dict[str, Tensor]]:
    """Forward pass + loss computation for an entire replay sequence batch.

    Orchestrates the full sequence-level world-model training pipeline:

    1. **Encode** — batch-encode all observations through the CNN.
    2. **RSSM loop** — step through time computing prior and posterior
       states at each step.
    3. **Decode & predict** — reconstruct observations, predict rewards,
       and (optionally) predict episode continuation from the stacked
       posterior features.
    4. **Loss** — call ``_compute_vectorized_losses`` for the core losses,
       then add optional overshooting regularization.
    5. **Aggregate** — combine all loss terms with their weights into
       ``total_loss``.

    Args:
        model:        The TinyWorldModel (encoder + RSSM + decoder + heads).
        observations: Input images, shape ``(B, T, C, H, W)``.
        actions:      Actions taken, shape ``(B, T, action_dim)``.
        rewards:      Scalar rewards, shape ``(B, T)``.
        dones:        Episode-done flags (fallback if ``terminals`` is None).
        terminals:    True-terminal flags, shape ``(B, T)``.
        kl_weight:    Multiplier for the KL divergence loss.
        free_nats:    KL floor (see ``_compute_vectorized_losses``).
        continue_loss_weight: Multiplier for the continue-predictor BCE.
        overshooting_horizon: Steps for latent overshooting (0 = disabled).
        overshooting_kl_weight: Multiplier for the overshooting KL.

    Returns:
        ``(outputs, losses)`` where ``outputs`` is a per-step list of
        ``WorldModelOutput`` and ``losses`` is a dict of named scalar
        tensors including ``total_loss``.
    """
    if observations.ndim != 5:
        raise ValueError("observations must have shape (B, T, C, H, W)")
    if actions.ndim != 3:
        raise ValueError("actions must have shape (B, T, action_dim)")
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (B, T)")

    terminal_targets = terminals if terminals is not None else dones
    batch_size, sequence_length = observations.shape[:2]

    # ── 1. Vectorized ENCODER ─────────────────────────────────────────
    # Flatten (B, T) into one batch, run CNN once, reshape back.
    # Result: (B, T, embedding_dim)
    embeddings = model.encoder.encode(observations)

    # ── 2. Recurrent RSSM Loop ────────────────────────────────────────
    # Step through each time position sequentially (GRU is inherently
    # sequential).  At each step we compute both the prior (imagine_step)
    # and the posterior (observe_step) so we can compute KL between them.
    prior_states: list[LatentState] = []
    posterior_states: list[LatentState] = []
    state = model.rssm.initial_state(batch_size, observations.device)

    for step in range(sequence_length):
        prior = model.rssm.imagine_step(state, actions[:, step])
        state = model.rssm.observe_step(state, actions[:, step], embeddings[:, step])
        prior_states.append(prior)
        posterior_states.append(state)

    # ── 3. Vectorized DECODER & REWARD ────────────────────────────────
    # Stack posterior features across time into (B, T, latent_dim) and
    # decode all steps at once, avoiding a Python loop.
    features = torch.stack([s.features for s in posterior_states], dim=1)

    reconstructions = model.decoder(features)                 # (B, T, C, H, W)
    predicted_rewards = model.reward_predictor(features)      # (B, T, 1)
    predicted_continues = (
        model.continue_predictor(features) if model.continue_predictor is not None else None
    )

    # ── 4. Extract Stacked Distributions for Loss ─────────────────────
    # Stack posterior and prior categorical logits into (B, T, num_cat, num_cls)
    # tensors for vectorized KL computation.
    post_logits = torch.stack([s.logits for s in posterior_states], dim=1)
    prior_logits = torch.stack([s.logits for s in prior_states], dim=1)

    # ── 5. Safe, Explicit Vectorized Loss Math (No Re-stacking) ───────
    losses = _compute_vectorized_losses(
        observations=observations,
        rewards=rewards,
        reconstructions=reconstructions,
        predicted_rewards=predicted_rewards,
        predicted_continues=predicted_continues,
        post_logits=post_logits,
        prior_logits=prior_logits,
        terminal_targets=terminal_targets,
        free_nats=free_nats,
        kl_balance=kl_balance,
        continue_loss_weight=continue_loss_weight,
        reward_std=(1.0 if model.reward_predictor.distribution_std is None else model.reward_predictor.distribution_std),
    )

    # ── 6. Reconstruct list[WorldModelOutput] to preserve API ─────────
    # Re-package per-step results into the WorldModelOutput struct so
    # that downstream consumers (e.g., overshooting, logging) can
    # access individual time-step posteriors and predictions.
    outputs: list[WorldModelOutput] = []
    for t in range(sequence_length):
        outputs.append(
            WorldModelOutput(
                embedding=embeddings[:, t],
                prior_state=prior_states[t],
                posterior_state=posterior_states[t],
                reconstruction=reconstructions[:, t],
                predicted_reward=predicted_rewards[:, t],
                predicted_observation_std=model.decoder.distribution_std,
                predicted_reward_std=model.reward_predictor.distribution_std,
                predicted_continue=predicted_continues[:, t] if predicted_continues is not None else None,
            )
        )

    # ── 7. Add Overshooting (Optional) ────────────────────────────────
    # Latent overshooting adds a regularization loss that penalizes
    # the prior for drifting from the posterior over multi-step rollouts.
    overshooting_losses = compute_latent_overshooting_losses(
        model, outputs, actions, overshooting_horizon=overshooting_horizon,
    )

    # ── 8. Aggregate total loss ───────────────────────────────────
    total_loss = (
        losses["reconstruction_loss"]
        + losses["reward_loss"]
        + kl_weight * losses["kl_loss"]
        + continue_loss_weight * losses["continue_loss"]
        + overshooting_kl_weight * overshooting_losses["overshooting_kl_loss"]
    )

    losses.update({
        "overshooting_kl_loss": overshooting_losses["overshooting_kl_loss"],
        "overshooting_feature_mse": overshooting_losses["overshooting_feature_mse"],
        "overshooting_pairs": overshooting_losses["overshooting_pairs"],
        "total_loss": total_loss,
    })

    return outputs, losses


def train_sequence_world_model_step(
    model: TinyWorldModel,
    optimizer: optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    rewards: Tensor,
    *,
    dones: Tensor | None = None,
    terminals: Tensor | None = None,
    kl_weight: float = 1.0,
    kl_balance: float = 0.8,
    free_nats: float = 3.0,
    continue_loss_weight: float = 1.0,
    overshooting_horizon: int = 0,
    overshooting_kl_weight: float = 0.0,
    grad_clip_norm: float = 100.0,
    grad_scaler: torch.amp.GradScaler | None = None,
    amp_context: torch.amp.autocast | None = None,
) -> tuple[list[WorldModelOutput], dict[str, float]]:
    """Single optimizer step: forward → loss → backward → clip → step.

    Wraps ``compute_sequence_world_model_losses`` with gradient management,
    optional AMP scaling, and gradient norm clipping.  Returns detached
    float metrics for logging.
    """
    optimizer.zero_grad(set_to_none=True)

    ctx = amp_context if amp_context is not None else nullcontext()
    with ctx:
        outputs, losses = compute_sequence_world_model_losses(
            model, observations, actions, rewards,
            dones=dones,
            terminals=terminals,
            kl_weight=kl_weight,
            kl_balance=kl_balance,
            free_nats=free_nats,
            continue_loss_weight=continue_loss_weight,
            overshooting_horizon=overshooting_horizon,
            overshooting_kl_weight=overshooting_kl_weight,
        )

    wm_grad_norm = _backward_and_step(
        losses["total_loss"], optimizer, model.parameters(),
        grad_clip_norm, grad_scaler,
    )
    metrics = {name: float(value.detach().item()) for name, value in losses.items()}
    metrics["wm_grad_norm"] = wm_grad_norm
    return outputs, metrics