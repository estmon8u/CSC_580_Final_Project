"""Tiny world-model loss and optimization helpers.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, optim

from tiny_dreamer_highway.models.world_model import TinyWorldModel, WorldModelOutput


def compute_world_model_losses(
    output: WorldModelOutput,
    target_observations: Tensor,
    target_rewards: Tensor,
) -> dict[str, Tensor]:
    observations_are_bytes = target_observations.dtype == torch.uint8
    target_observations = target_observations.to(dtype=output.reconstruction.dtype)
    if observations_are_bytes:
        target_observations = target_observations / 255.0

    reward_targets = target_rewards.reshape(-1, 1).to(dtype=output.predicted_reward.dtype)
    reconstruction_loss = F.mse_loss(output.reconstruction, target_observations)
    reward_loss = F.mse_loss(output.predicted_reward, reward_targets)
    total_loss = reconstruction_loss + reward_loss
    return {
        "reconstruction_loss": reconstruction_loss,
        "reward_loss": reward_loss,
        "total_loss": total_loss,
    }


def train_world_model_step(
    model: TinyWorldModel,
    optimizer: optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    rewards: Tensor,
) -> tuple[WorldModelOutput, dict[str, float]]:
    optimizer.zero_grad(set_to_none=True)
    output = model(observations, actions)
    losses = compute_world_model_losses(output, observations, rewards)
    losses["total_loss"].backward()
    optimizer.step()
    return output, {name: float(value.detach().cpu().item()) for name, value in losses.items()}