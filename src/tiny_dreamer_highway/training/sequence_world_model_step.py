"""Sequence training helpers for the tiny world model.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor, optim

from tiny_dreamer_highway.models.encoder import LatentState
from tiny_dreamer_highway.models.world_model import TinyWorldModel, WorldModelOutput
from tiny_dreamer_highway.training.world_model_step import (
    _backward_and_step,
    compute_world_model_losses,
)
from tiny_dreamer_highway.types import ReplaySequenceBatch, Transition


def stack_sequence_batch(sequences: list[list[Transition]]) -> ReplaySequenceBatch:
    if not sequences or not sequences[0]:
        raise ValueError("sequences must be non-empty")

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    for sequence in sequences:
        observations.append([transition.observation for transition in sequence])
        actions.append([transition.action for transition in sequence])
        rewards.append([transition.reward for transition in sequence])
        next_observations.append([transition.next_observation for transition in sequence])
        dones.append([transition.done for transition in sequence])

    return ReplaySequenceBatch(
        observations=np.asarray(observations, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.uint8),
        dones=np.asarray(dones, dtype=np.bool_),
    )


def compute_sequence_world_model_losses(
    model: TinyWorldModel,
    observations: Tensor,
    actions: Tensor,
    rewards: Tensor,
    *,
    dones: Tensor | None = None,
    kl_weight: float = 1.0,
    free_nats: float = 3.0,
    continue_loss_weight: float = 1.0,
) -> tuple[list[WorldModelOutput], dict[str, Tensor]]:
    if observations.ndim != 5:
        raise ValueError("observations must have shape (B, T, C, H, W)")
    if actions.ndim != 3:
        raise ValueError("actions must have shape (B, T, action_dim)")
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (B, T)")

    batch_size, sequence_length = observations.shape[:2]
    state: LatentState | None = None
    outputs: list[WorldModelOutput] = []
    reconstruction_loss = torch.zeros((), device=observations.device)
    reconstruction_mse = torch.zeros((), device=observations.device)
    observation_log_prob = torch.zeros((), device=observations.device)
    reward_loss = torch.zeros((), device=observations.device)
    continue_loss = torch.zeros((), device=observations.device)
    kl_loss = torch.zeros((), device=observations.device)
    kl_loss_raw = torch.zeros((), device=observations.device)

    for step in range(sequence_length):
        output = model(observations[:, step], actions[:, step], prev_state=state)
        step_losses = compute_world_model_losses(
            output,
            observations[:, step],
            rewards[:, step],
            target_dones=None if dones is None else dones[:, step],
            kl_weight=1.0, free_nats=free_nats,
            continue_loss_weight=continue_loss_weight,
        )
        reconstruction_loss = reconstruction_loss + step_losses["reconstruction_loss"]
        reconstruction_mse = reconstruction_mse + step_losses["reconstruction_mse"]
        observation_log_prob = observation_log_prob + step_losses["observation_log_prob"]
        reward_loss = reward_loss + step_losses["reward_loss"]
        continue_loss = continue_loss + step_losses["continue_loss"]
        kl_loss = kl_loss + step_losses["kl_loss"]
        kl_loss_raw = kl_loss_raw + step_losses["kl_loss_raw"]
        outputs.append(output)
        state = output.posterior_state

    reconstruction_loss = reconstruction_loss / sequence_length
    reconstruction_mse = reconstruction_mse / sequence_length
    observation_log_prob = observation_log_prob / sequence_length
    reward_loss = reward_loss / sequence_length
    continue_loss = continue_loss / sequence_length
    kl_loss = kl_loss / sequence_length
    kl_loss_raw = kl_loss_raw / sequence_length
    total_loss = (
        reconstruction_loss
        + reward_loss
        + kl_weight * kl_loss
        + continue_loss_weight * continue_loss
    )
    return outputs, {
        "reconstruction_loss": reconstruction_loss,
        "reconstruction_mse": reconstruction_mse,
        "observation_log_prob": observation_log_prob,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
        "kl_loss_raw": kl_loss_raw.detach(),
        "total_loss": total_loss,
    }


def train_sequence_world_model_step(
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
) -> tuple[list[WorldModelOutput], dict[str, float]]:
    optimizer.zero_grad(set_to_none=True)

    ctx = amp_context if amp_context is not None else nullcontext()
    with ctx:
        outputs, losses = compute_sequence_world_model_losses(
            model, observations, actions, rewards,
            dones=dones,
            kl_weight=kl_weight,
            free_nats=free_nats,
            continue_loss_weight=continue_loss_weight,
        )

    _backward_and_step(
        losses["total_loss"], optimizer, model.parameters(),
        grad_clip_norm, grad_scaler,
    )
    return outputs, {name: float(value.detach().cpu().item()) for name, value in losses.items()}