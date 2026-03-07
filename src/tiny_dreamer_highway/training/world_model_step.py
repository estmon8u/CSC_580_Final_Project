"""Tiny world-model loss and optimization helpers.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torch import Tensor, optim

from tiny_dreamer_highway.models.world_model import TinyWorldModel, WorldModelOutput


# ---------------------------------------------------------------------------
# Gaussian KL divergence — core Dreamer V1 regularisation
# ---------------------------------------------------------------------------

def gaussian_kl_divergence(
    posterior_mean: Tensor,
    posterior_std: Tensor,
    prior_mean: Tensor,
    prior_std: Tensor,
) -> Tensor:
    """Analytic KL(posterior || prior) for diagonal Gaussians.

    Returns a scalar (mean over batch and latent dimensions).
    """
    var_ratio = (posterior_std / prior_std).pow(2)
    mean_diff = ((prior_mean - posterior_mean) / prior_std).pow(2)
    kl_per_dim = 0.5 * (var_ratio + mean_diff - 1.0 - var_ratio.log())
    return kl_per_dim.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_world_model_losses(
    output: WorldModelOutput,
    target_observations: Tensor,
    target_rewards: Tensor,
    *,
    kl_weight: float = 1.0,
    free_nats: float = 3.0,
    target_dones: Tensor | None = None,
    continue_loss_weight: float = 1.0,
) -> dict[str, Tensor]:
    observations_are_bytes = target_observations.dtype == torch.uint8
    target_observations = target_observations.to(dtype=output.reconstruction.dtype)
    if observations_are_bytes:
        target_observations = target_observations / 255.0

    reward_targets = target_rewards.reshape(-1, 1).to(dtype=output.predicted_reward.dtype)
    reconstruction_loss = F.mse_loss(output.reconstruction, target_observations)
    reward_std = 1.0 if output.predicted_reward_std is None else output.predicted_reward_std
    reward_dist = Independent(
        Normal(output.predicted_reward, torch.full_like(output.predicted_reward, reward_std)),
        1,
    )
    reward_loss = -reward_dist.log_prob(reward_targets).mean()
    continue_loss = torch.zeros((), device=target_observations.device, dtype=reward_loss.dtype)
    if output.predicted_continue is not None and target_dones is not None:
        continue_targets = (1.0 - target_dones.reshape(-1, 1).to(dtype=output.predicted_continue.dtype))
        continue_loss = F.binary_cross_entropy_with_logits(
            output.predicted_continue,
            continue_targets,
        )

    # KL divergence between posterior and prior (Dreamer V1 §3)
    posterior = output.posterior_state
    prior = output.prior_state
    if (
        posterior.dist_mean is not None
        and posterior.dist_std is not None
        and prior.dist_mean is not None
        and prior.dist_std is not None
    ):
        raw_kl = gaussian_kl_divergence(
            posterior.dist_mean,
            posterior.dist_std,
            prior.dist_mean,
            prior.dist_std,
        )
        # Free-nats: clamp KL below a threshold so the model does not over-
        # regularise early in training (Dreamer V1 default = 3 nats).
        kl_loss = torch.clamp(raw_kl, min=free_nats)
    else:
        raw_kl = torch.zeros((), device=target_observations.device)
        kl_loss = raw_kl

    total_loss = (
        reconstruction_loss
        + reward_loss
        + kl_weight * kl_loss
        + continue_loss_weight * continue_loss
    )
    return {
        "reconstruction_loss": reconstruction_loss,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
        "kl_loss_raw": raw_kl.detach(),
        "total_loss": total_loss,
    }


# ---------------------------------------------------------------------------
# Backward + clip + step  (shared helper)
# ---------------------------------------------------------------------------

def _backward_and_step(
    loss: Tensor,
    optimizer: optim.Optimizer,
    parameters,
    grad_clip_norm: float,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Backward pass, gradient clipping, and optimizer step.

    Supports an optional :class:`GradScaler` for float16 AMP.
    """
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip_norm)
        optimizer.step()


# ---------------------------------------------------------------------------
# Single-step training helper
# ---------------------------------------------------------------------------

def train_world_model_step(
    model: TinyWorldModel,
    optimizer: optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    rewards: Tensor,
    *,
    dones: Tensor | None = None,
    kl_weight: float = 1.0,
    free_nats: float = 3.0,
    continue_loss_weight: float = 1.0,
    grad_clip_norm: float = 100.0,
    grad_scaler: torch.amp.GradScaler | None = None,
    amp_context: torch.amp.autocast | None = None,
) -> tuple[WorldModelOutput, dict[str, float]]:
    optimizer.zero_grad(set_to_none=True)

    ctx = amp_context if amp_context is not None else nullcontext()
    with ctx:
        output = model(observations, actions)
        losses = compute_world_model_losses(
            output,
            observations,
            rewards,
            kl_weight=kl_weight,
            free_nats=free_nats,
            target_dones=dones,
            continue_loss_weight=continue_loss_weight,
        )

    _backward_and_step(
        losses["total_loss"], optimizer, model.parameters(),
        grad_clip_norm, grad_scaler,
    )
    return output, {name: float(value.detach().cpu().item()) for name, value in losses.items()}