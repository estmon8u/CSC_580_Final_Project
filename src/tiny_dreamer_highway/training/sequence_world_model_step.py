"""Sequence training helpers for the tiny world model.

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
    gaussian_kl_divergence,
)
from tiny_dreamer_highway.types import ReplaySequenceBatch, Transition


def _compute_vectorized_losses(
    observations: Tensor,
    rewards: Tensor,
    reconstructions: Tensor,
    predicted_rewards: Tensor,
    predicted_continues: Tensor | None,
    post_means: Tensor,
    post_stds: Tensor,
    prior_means: Tensor,
    prior_stds: Tensor,
    *,
    terminal_targets: Tensor | None,
    free_nats: float,
    continue_loss_weight: float,
    observation_std: float,
    reward_std: float,
) -> dict[str, Tensor]:
    """Compute all world-model losses in one vectorized pass.

    Accepts pre-stacked (B, T, ...) tensors directly to avoid Python overhead.
    """
    B, T = observations.shape[:2]
    device = observations.device

    # ── Target preparation ───────────────────────────────────────────
    target_obs = observations.to(dtype=reconstructions.dtype)
    if observations.dtype == torch.uint8:
        target_obs = target_obs / 255.0
    target_rew = rewards.unsqueeze(-1).to(dtype=predicted_rewards.dtype)  # (B, T, 1)

    # ── Reconstruction loss (Gaussian log-prob) ──────────────────────
    recon_var = observation_std ** 2
    recon_sq_err = (reconstructions - target_obs).pow(2)
    reconstruction_mse = recon_sq_err.mean()

    event_dims = reconstructions.shape[2:]  # (C, H, W)
    d = 1
    for s in event_dims:
        d *= s

    log_prob_per_sample = -0.5 * (
        d * math.log(2 * math.pi)
        + d * math.log(recon_var)
        + recon_sq_err.reshape(B, T, -1).sum(dim=-1) / recon_var
    )  # (B, T)
    observation_log_prob = log_prob_per_sample.mean()
    reconstruction_loss = -observation_log_prob

    # ── Reward loss (Gaussian log-prob) ──────────────────────────────
    rew_sq_err = (predicted_rewards - target_rew).pow(2)
    rew_var = reward_std ** 2
    rew_log_prob = -0.5 * (
        math.log(2 * math.pi) + math.log(rew_var) + rew_sq_err.squeeze(-1) / rew_var
    )  # (B, T)
    reward_loss = -rew_log_prob.mean()

    # ── Continue loss (BCE) ──────────────────────────────────────────
    continue_loss = torch.zeros((), device=device, dtype=reconstructions.dtype)
    if terminal_targets is not None and predicted_continues is not None:
        continue_targets = (
            1.0 - terminal_targets.unsqueeze(-1).to(dtype=predicted_continues.dtype)
        )  # (B, T, 1)
        continue_loss = F.binary_cross_entropy_with_logits(
            predicted_continues, continue_targets,
        )

    # ── KL divergence (analytic Gaussian KL) ─────────────────────────
    var_ratio = (post_stds / prior_stds).pow(2)
    mean_diff = ((prior_means - post_means) / prior_stds).pow(2)
    kl_per_dim = 0.5 * (var_ratio + mean_diff - 1.0 - var_ratio.log())

    raw_kl = kl_per_dim.sum(dim=-1).mean()
    kl_loss = torch.clamp(raw_kl, min=free_nats)

    return {
        "reconstruction_loss": reconstruction_loss,
        "reconstruction_mse": reconstruction_mse,
        "observation_log_prob": observation_log_prob,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
        "kl_loss_raw": raw_kl.detach(),
    }


def compute_latent_overshooting_losses(
    model: TinyWorldModel,
    outputs: list[WorldModelOutput],
    actions: Tensor,
    *,
    overshooting_horizon: int,
) -> dict[str, Tensor]:
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
        rollout_actions = actions[:, start_index + 1 : start_index + 1 + rollout_horizon]
        detached_start_state = LatentState(
            deterministic=start_state.deterministic.detach(),
            stochastic=start_state.stochastic.detach(),
        )
        imagined_states = model.rssm.imagine_rollout(detached_start_state, rollout_actions)

        for offset, imagined_state in enumerate(imagined_states, start=1):
            target_state = outputs[start_index + offset].posterior_state
            if (
                imagined_state.dist_mean is None
                or imagined_state.dist_std is None
                or target_state.dist_mean is None
                or target_state.dist_std is None
                or target_state.deterministic is None
                or target_state.stochastic is None
            ):
                continue

            kl_total = kl_total + gaussian_kl_divergence(
                target_state.dist_mean.detach(),
                target_state.dist_std.detach(),
                imagined_state.dist_mean,
                imagined_state.dist_std,
            )
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
    free_nats: float = 3.0,
    continue_loss_weight: float = 1.0,
    overshooting_horizon: int = 0,
    overshooting_kl_weight: float = 0.0,
) -> tuple[list[WorldModelOutput], dict[str, Tensor]]:
    if observations.ndim != 5:
        raise ValueError("observations must have shape (B, T, C, H, W)")
    if actions.ndim != 3:
        raise ValueError("actions must have shape (B, T, action_dim)")
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (B, T)")

    terminal_targets = terminals if terminals is not None else dones
    batch_size, sequence_length = observations.shape[:2]

    # ── 1. Vectorized ENCODER ─────────────────────────────────────────
    embeddings = model.encoder.encode(observations)

    # ── 2. Recurrent RSSM Loop ────────────────────────────────────────
    prior_states: list[LatentState] = []
    posterior_states: list[LatentState] = []
    state = model.rssm.initial_state(batch_size, observations.device)

    for step in range(sequence_length):
        prior = model.rssm.imagine_step(state, actions[:, step])
        state = model.rssm.observe_step(state, actions[:, step], embeddings[:, step])
        prior_states.append(prior)
        posterior_states.append(state)

    # ── 3. Vectorized DECODER & REWARD ────────────────────────────────
    # Stack features to compute all time steps in one GPU kernel
    features = torch.stack([s.features for s in posterior_states], dim=1)

    reconstructions = model.decoder(features)                 # (B, T, C, H, W)
    predicted_rewards = model.reward_predictor(features)      # (B, T, 1)
    predicted_continues = (
        model.continue_predictor(features) if model.continue_predictor is not None else None
    )

    # ── 4. Extract Stacked Distributions for Loss ─────────────────────
    post_means = torch.stack([s.dist_mean for s in posterior_states], dim=1)
    post_stds = torch.stack([s.dist_std for s in posterior_states], dim=1)
    prior_means = torch.stack([s.dist_mean for s in prior_states], dim=1)
    prior_stds = torch.stack([s.dist_std for s in prior_states], dim=1)

    # ── 5. Safe, Explicit Vectorized Loss Math (No Re-stacking) ───────
    losses = _compute_vectorized_losses(
        observations=observations,
        rewards=rewards,
        reconstructions=reconstructions,
        predicted_rewards=predicted_rewards,
        predicted_continues=predicted_continues,
        post_means=post_means,
        post_stds=post_stds,
        prior_means=prior_means,
        prior_stds=prior_stds,
        terminal_targets=terminal_targets,
        free_nats=free_nats,
        continue_loss_weight=continue_loss_weight,
        observation_std=(1.0 if model.decoder.distribution_std is None else model.decoder.distribution_std),
        reward_std=(1.0 if model.reward_predictor.distribution_std is None else model.reward_predictor.distribution_std),
    )

    # ── 6. Reconstruct list[WorldModelOutput] to preserve API ─────────
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
    overshooting_losses = compute_latent_overshooting_losses(
        model, outputs, actions, overshooting_horizon=overshooting_horizon,
    )

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
    free_nats: float = 3.0,
    continue_loss_weight: float = 1.0,
    overshooting_horizon: int = 0,
    overshooting_kl_weight: float = 0.0,
    grad_clip_norm: float = 100.0,
    grad_scaler: torch.amp.GradScaler | None = None,
    amp_context: torch.amp.autocast | None = None,
) -> tuple[list[WorldModelOutput], dict[str, float]]:
    optimizer.zero_grad(set_to_none=True)

    ctx = amp_context if amp_context is not None else nullcontext()
    with ctx:
        outputs, losses = compute_sequence_world_model_losses(
            model, observations, actions, rewards,
            dones=dones,
            terminals=terminals,
            kl_weight=kl_weight,
            free_nats=free_nats,
            continue_loss_weight=continue_loss_weight,
            overshooting_horizon=overshooting_horizon,
            overshooting_kl_weight=overshooting_kl_weight,
        )

    _backward_and_step(
        losses["total_loss"], optimizer, model.parameters(),
        grad_clip_norm, grad_scaler,
    )
    return outputs, {name: float(value.detach().item()) for name, value in losses.items()}