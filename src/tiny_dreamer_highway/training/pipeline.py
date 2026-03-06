"""Minimal alternating training pipeline helpers.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, optim

from tiny_dreamer_highway.config import ExperimentConfig, TrainingConfig
from tiny_dreamer_highway.data.collect_random_rollouts import collect_random_transitions
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.envs.highway_factory import make_highway_env
from tiny_dreamer_highway.models import Actor, LatentState, TinyWorldModel, Critic
from tiny_dreamer_highway.training.behavior_learning import train_behavior_step
from tiny_dreamer_highway.training.world_model_step import train_world_model_step
from tiny_dreamer_highway.types import Transition


@dataclass(slots=True)
class PipelineCycleMetrics:
    warm_start_added: int
    policy_added: int
    replay_size: int
    world_model_metrics: dict[str, float]
    behavior_metrics: dict[str, float]


def _observation_to_tensor(observation: np.ndarray) -> Tensor:
    observation_tensor = torch.as_tensor(observation)
    if observation_tensor.ndim == 2:
        observation_tensor = observation_tensor.unsqueeze(0)
    if observation_tensor.ndim == 3:
        observation_tensor = observation_tensor.unsqueeze(0)
    return observation_tensor


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def seed_latent_state(
    world_model: TinyWorldModel,
    observations: Tensor,
    actions: Tensor,
) -> LatentState:
    with torch.no_grad():
        output = world_model(observations, actions)
    return output.posterior_state


def collect_actor_transitions(
    config: ExperimentConfig,
    replay_buffer: ReplayBuffer,
    world_model: TinyWorldModel,
    actor: Actor,
    steps: int,
    seed: int | None = None,
) -> int:
    if steps <= 0:
        return 0

    env = make_highway_env(config.env)
    if seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    model_device = _module_device(world_model)
    prev_state = world_model.rssm.initial_state(batch_size=1, device=model_device)
    added = 0

    try:
        for _ in range(steps):
            with torch.no_grad():
                action_tensor = actor(prev_state.features)
                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                observation_tensor = _observation_to_tensor(
                    np.asarray(observation, dtype=np.uint8)
                ).to(model_device)
                posterior = world_model(observation_tensor, action_tensor, prev_state=prev_state)
                prev_state = posterior.posterior_state

            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay_buffer.add(
                Transition(
                    observation=np.asarray(observation, dtype=np.uint8),
                    action=action,
                    reward=float(reward),
                    next_observation=np.asarray(next_observation, dtype=np.uint8),
                    done=done,
                )
            )
            added += 1
            observation = next_observation

            if done:
                observation, _ = env.reset()
                prev_state = world_model.rssm.initial_state(batch_size=1, device=model_device)
    finally:
        env.close()

    return added


def run_training_cycle(
    config: ExperimentConfig,
    replay_buffer: ReplayBuffer,
    world_model: TinyWorldModel,
    actor: Actor,
    critic: Critic,
    world_model_optimizer: optim.Optimizer,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    warm_start_steps: int = 0,
    policy_steps: int = 0,
    seed: int | None = None,
) -> PipelineCycleMetrics:
    warm_start_added = 0
    if warm_start_steps > 0:
        warm_start_added = collect_random_transitions(
            config.env,
            replay_buffer,
            steps=warm_start_steps,
            seed=seed,
        )

    batch_size = config.training.batch_size
    if not replay_buffer.can_sample(batch_size=batch_size):
        raise ValueError("replay buffer does not contain enough samples for a training cycle")

    batch = replay_buffer.sample_batch(batch_size=batch_size)
    model_device = _module_device(world_model)
    observations = torch.as_tensor(batch.observations, device=model_device)
    actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=model_device)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=model_device)

    _, world_model_metrics = train_world_model_step(
        world_model,
        world_model_optimizer,
        observations,
        actions,
        rewards,
    )

    start_state = seed_latent_state(world_model, observations, actions)
    training_config: TrainingConfig = config.training
    behavior_metrics = train_behavior_step(
        world_model,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        start_state,
        horizon=training_config.imagination_horizon,
    )

    policy_added = collect_actor_transitions(
        config,
        replay_buffer,
        world_model,
        actor,
        steps=policy_steps,
        seed=seed,
    )

    return PipelineCycleMetrics(
        warm_start_added=warm_start_added,
        policy_added=policy_added,
        replay_size=len(replay_buffer),
        world_model_metrics=world_model_metrics,
        behavior_metrics=behavior_metrics,
    )