"""Imagination and actor-critic training helpers.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from tiny_dreamer_highway.models import Actor, Critic, LatentState, TinyWorldModel
from tiny_dreamer_highway.utils import stabilize_action_tensor


@dataclass(slots=True)
class ImaginedTrajectory:
    states: list[LatentState]
    features: Tensor
    actions: Tensor
    rewards: Tensor
    values: Tensor
    bootstrap: Tensor


def _noop_context():
    """Return a context manager that does nothing (for non-AMP paths)."""
    return nullcontext()


@contextmanager
def frozen_params(module: nn.Module) -> Iterator[None]:
    original_flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original_flags, strict=True):
            parameter.requires_grad_(requires_grad)


def imagine_trajectory(
    world_model: TinyWorldModel,
    actor: Actor,
    critic: Critic,
    start_state: LatentState,
    horizon: int,
    *,
    longitudinal_scale: float = 1.0,
    lateral_scale: float = 1.0,
    smoothing_factor: float = 0.0,
    lateral_control: bool = True,
) -> ImaginedTrajectory:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if start_state.deterministic is None or start_state.stochastic is None:
        raise ValueError("start_state must contain deterministic and stochastic tensors")

    state = start_state
    states: list[LatentState] = []
    feature_steps: list[Tensor] = []
    action_steps: list[Tensor] = []
    reward_steps: list[Tensor] = []
    value_steps: list[Tensor] = []
    previous_action: Tensor | None = None

    for _ in range(horizon):
        action = stabilize_action_tensor(
            actor(state.features),
            previous_action=previous_action,
            longitudinal_scale=longitudinal_scale,
            lateral_scale=lateral_scale,
            smoothing_factor=smoothing_factor,
            lateral_enabled=lateral_control,
        )
        state = world_model.rssm.imagine_step(state, action)
        features = state.features
        reward = world_model.reward_predictor(features)
        value = critic(features)

        states.append(state)
        feature_steps.append(features)
        action_steps.append(action)
        reward_steps.append(reward)
        value_steps.append(value)
        previous_action = action

    stacked_features = torch.stack(feature_steps, dim=0)
    stacked_actions = torch.stack(action_steps, dim=0)
    stacked_rewards = torch.stack(reward_steps, dim=0)
    stacked_values = torch.stack(value_steps, dim=0)
    bootstrap = critic(state.features)
    return ImaginedTrajectory(
        states=states,
        features=stacked_features,
        actions=stacked_actions,
        rewards=stacked_rewards,
        values=stacked_values,
        bootstrap=bootstrap,
    )


def td_lambda_returns(
    rewards: Tensor,
    values: Tensor,
    bootstrap: Tensor | None = None,
    discount: float = 0.99,
    lambda_: float = 0.95,
) -> Tensor:
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have matching shapes")
    if rewards.ndim < 2:
        raise ValueError("rewards and values must include time and batch dimensions")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    if bootstrap is None:
        bootstrap = values[-1]

    returns = torch.zeros_like(rewards)
    next_return = bootstrap
    for step in range(rewards.shape[0] - 1, -1, -1):
        blended_target = (1.0 - lambda_) * values[step] + lambda_ * next_return
        next_return = rewards[step] + discount * blended_target
        returns[step] = next_return
    return returns


def train_behavior_step(
    world_model: TinyWorldModel,
    actor: Actor,
    critic: Critic,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    start_state: LatentState,
    horizon: int,
    discount: float = 0.99,
    lambda_: float = 0.95,
    grad_clip_norm: float = 100.0,
    longitudinal_scale: float = 1.0,
    lateral_scale: float = 1.0,
    smoothing_factor: float = 0.0,
    lateral_control: bool = True,
    actor_scaler: torch.amp.GradScaler | None = None,
    critic_scaler: torch.amp.GradScaler | None = None,
    amp_context: torch.amp.autocast | None = None,
) -> dict[str, float]:
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)

    _amp = amp_context if amp_context is not None else _noop_context()
    with _amp, frozen_params(world_model), frozen_params(critic):
        imagined_for_actor = imagine_trajectory(
            world_model,
            actor,
            critic,
            start_state,
            horizon,
            longitudinal_scale=longitudinal_scale,
            lateral_scale=lateral_scale,
            smoothing_factor=smoothing_factor,
            lateral_control=lateral_control,
        )
        actor_returns = td_lambda_returns(
            imagined_for_actor.rewards,
            imagined_for_actor.values,
            bootstrap=imagined_for_actor.bootstrap,
            discount=discount,
            lambda_=lambda_,
        )
        actor_loss = -actor_returns.mean()

    if actor_scaler is not None:
        actor_scaler.scale(actor_loss).backward()
        actor_scaler.unscale_(actor_optimizer)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=grad_clip_norm)
        actor_scaler.step(actor_optimizer)
        actor_scaler.update()
    else:
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=grad_clip_norm)
        actor_optimizer.step()

    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)

    with _amp, frozen_params(world_model), frozen_params(actor):
        imagined_for_critic = imagine_trajectory(
            world_model,
            actor,
            critic,
            start_state,
            horizon,
            longitudinal_scale=longitudinal_scale,
            lateral_scale=lateral_scale,
            smoothing_factor=smoothing_factor,
            lateral_control=lateral_control,
        )
        critic_targets = td_lambda_returns(
            imagined_for_critic.rewards,
            imagined_for_critic.values,
            bootstrap=imagined_for_critic.bootstrap,
            discount=discount,
            lambda_=lambda_,
        ).detach()
        critic_loss = F.mse_loss(imagined_for_critic.values, critic_targets)

    if critic_scaler is not None:
        critic_scaler.scale(critic_loss).backward()
        critic_scaler.unscale_(critic_optimizer)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip_norm)
        critic_scaler.step(critic_optimizer)
        critic_scaler.update()
    else:
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip_norm)
        critic_optimizer.step()

    metrics = {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "imagined_reward_mean": float(imagined_for_critic.rewards.detach().mean().cpu().item()),
        "imagined_value_mean": float(imagined_for_critic.values.detach().mean().cpu().item()),
    }
    return metrics