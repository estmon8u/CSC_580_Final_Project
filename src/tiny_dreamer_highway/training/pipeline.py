"""Minimal alternating training pipeline helpers.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor, optim

from tiny_dreamer_highway.config import ExperimentConfig, TrainingConfig
from tiny_dreamer_highway.data.collect_random_rollouts import collect_random_transitions
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.envs.highway_factory import make_highway_env
from tiny_dreamer_highway.models import Actor, LatentState, TinyWorldModel, Critic
from tiny_dreamer_highway.training.behavior_learning import train_behavior_step
from tiny_dreamer_highway.training.sequence_world_model_step import (
    stack_sequence_batch,
    train_sequence_world_model_step,
)
from tiny_dreamer_highway.training.world_model_step import train_world_model_step
from tiny_dreamer_highway.types import Transition
from tiny_dreamer_highway.utils import stabilize_action_tensor


@dataclass(slots=True)
class PipelineCycleMetrics:
    warm_start_added: int
    policy_added: int
    replay_size: int
    world_model_metrics: dict[str, float]
    behavior_metrics: dict[str, float]
    evaluation_metrics: dict[str, float] = field(default_factory=dict)


def resolve_amp_dtype(name: str) -> torch.dtype:
    """Map config string to torch dtype for AMP."""
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _average_metric_dicts(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {
        key: float(sum(metrics[key] for metrics in metrics_list) / len(metrics_list))
        for key in keys
    }


def _observation_to_tensor(observation: np.ndarray) -> Tensor:
    observation_tensor = torch.as_tensor(observation)
    if observation_tensor.ndim == 2:
        observation_tensor = observation_tensor.unsqueeze(0)
    if observation_tensor.ndim == 3:
        observation_tensor = observation_tensor.unsqueeze(0)
    return observation_tensor


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _ensure_sampleable_replay_sequences(
    config: ExperimentConfig,
    replay_buffer: ReplayBuffer,
    *,
    batch_size: int,
    sequence_length: int,
    seed: int | None,
) -> int:
    if replay_buffer.can_sample(batch_size=batch_size, sequence_length=sequence_length):
        return 0

    max_episode_steps = int(config.env.max_episode_steps)
    if sequence_length > max_episode_steps:
        raise ValueError(
            "replay sequence sampling is impossible with the current configuration: "
            f"sequence_length={sequence_length} exceeds max_episode_steps={max_episode_steps}. "
            "Reduce sequence_length or increase the episode horizon before training."
        )

    collection_chunk = max(sequence_length, min(max_episode_steps, batch_size * sequence_length))
    max_extra_steps = max(collection_chunk * 4, max_episode_steps * 4)
    extra_added = 0
    attempt = 0

    while (
        extra_added < max_extra_steps
        and not replay_buffer.can_sample(batch_size=batch_size, sequence_length=sequence_length)
    ):
        remaining = max_extra_steps - extra_added
        steps = min(collection_chunk, remaining)
        if steps <= 0:
            break
        extra_added += collect_random_transitions(
            config.env,
            replay_buffer,
            steps=steps,
            seed=None if seed is None else seed + 10_000 + attempt,
        )
        attempt += 1

    return extra_added


def seed_latent_state(
    world_model: TinyWorldModel,
    observations: Tensor,
    actions: Tensor,
    *,
    amp_context: torch.amp.autocast | None = None,
) -> LatentState:
    ctx = amp_context if amp_context is not None else nullcontext()
    with torch.no_grad(), ctx:
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
    action_dim = world_model.rssm.action_dim
    prev_state = world_model.rssm.initial_state(batch_size=1, device=model_device)
    prev_action = torch.zeros(1, action_dim, device=model_device)
    added = 0

    try:
        for _ in range(steps):
            with torch.no_grad():
                # 1. Encode current observation → posterior (uses previous action for GRU)
                observation_tensor = _observation_to_tensor(
                    np.asarray(observation, dtype=np.uint8)
                ).to(model_device)
                posterior = world_model(observation_tensor, prev_action, prev_state=prev_state)
                prev_state = posterior.posterior_state

                # 2. Select action based on the posterior that sees current obs
                action_tensor = stabilize_action_tensor(
                    actor(prev_state.features),
                    previous_action=prev_action,
                    longitudinal_scale=config.env.action.longitudinal_scale,
                    lateral_scale=config.env.action.lateral_scale,
                    smoothing_factor=config.env.action.smoothing_factor,
                    lateral_enabled=config.env.action.lateral,
                )
                prev_action = action_tensor
                action = action_tensor.squeeze(0).float().cpu().numpy()

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
                prev_action = torch.zeros(1, action_dim, device=model_device)
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
    wm_scaler: torch.amp.GradScaler | None = None,
    actor_scaler: torch.amp.GradScaler | None = None,
    critic_scaler: torch.amp.GradScaler | None = None,
    amp_context: torch.amp.autocast | None = None,
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
    sequence_length = config.replay.sequence_length
    warm_start_added += _ensure_sampleable_replay_sequences(
        config,
        replay_buffer,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    if not replay_buffer.can_sample(batch_size=batch_size, sequence_length=sequence_length):
        valid_sequences = len(replay_buffer.valid_sequence_start_indices(sequence_length))
        raise ValueError(
            "replay buffer does not yet contain enough valid contiguous training sequences "
            f"(sequence_length={sequence_length}, replay_size={len(replay_buffer)}, "
            f"valid_sequences={valid_sequences}, warm_start_added={warm_start_added}). "
            "The trainer already tried topping up random warm-start data. Increase warm_start_steps, "
            "reduce sequence_length/batch_size for short validation runs, or relax terminal settings "
            "so more episodes survive long enough to produce contiguous replay windows."
        )

    training_config: TrainingConfig = config.training
    model_device = _module_device(world_model)

    world_model_metrics_list: list[dict[str, float]] = []
    # Collect posterior states from WM training for behavior update start states
    all_posterior_states: list[LatentState] = []
    for _ in range(training_config.world_model_updates_per_cycle):
        sequences = replay_buffer.sample_sequences(
            batch_size=batch_size, sequence_length=sequence_length,
        )
        seq_batch = stack_sequence_batch(sequences)
        observations = torch.as_tensor(seq_batch.observations, device=model_device)
        actions = torch.as_tensor(seq_batch.actions, dtype=torch.float32, device=model_device)
        rewards = torch.as_tensor(seq_batch.rewards, dtype=torch.float32, device=model_device)

        outputs, world_model_metrics = train_sequence_world_model_step(
            world_model,
            world_model_optimizer,
            observations,
            actions,
            rewards,
            dones=torch.as_tensor(seq_batch.dones, dtype=torch.float32, device=model_device),
            kl_weight=training_config.kl_weight,
            free_nats=training_config.free_nats,
            continue_loss_weight=training_config.continue_loss_weight,
            overshooting_horizon=training_config.overshooting_horizon,
            overshooting_kl_weight=training_config.overshooting_kl_weight,
            grad_clip_norm=training_config.grad_clip_norm,
            grad_scaler=wm_scaler,
            amp_context=amp_context,
        )
        world_model_metrics_list.append(world_model_metrics)
        # Gather detached posteriors from all time steps
        for wm_output in outputs:
            all_posterior_states.append(
                LatentState(
                    deterministic=wm_output.posterior_state.deterministic.detach(),
                    stochastic=wm_output.posterior_state.stochastic.detach(),
                )
            )

    behavior_metrics_list: list[dict[str, float]] = []
    for _ in range(training_config.behavior_updates_per_cycle):
        # Sample start states from collected WM posteriors
        if all_posterior_states:
            # Flatten all posteriors: each entry is (B, dim), concatenate across time & batches
            all_det = torch.cat([s.deterministic for s in all_posterior_states], dim=0)
            all_sto = torch.cat([s.stochastic for s in all_posterior_states], dim=0)
            # Random subsample to batch_size
            n_total = all_det.shape[0]
            indices = torch.randint(0, n_total, (batch_size,), device=model_device)
            start_state = LatentState(
                deterministic=all_det[indices],
                stochastic=all_sto[indices],
            )
        else:
            # Fallback: seed from replay buffer
            batch = replay_buffer.sample_batch(batch_size=batch_size)
            observations = torch.as_tensor(batch.observations, device=model_device)
            actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=model_device)
            start_state = seed_latent_state(world_model, observations, actions, amp_context=amp_context)
        behavior_metrics = train_behavior_step(
            world_model,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            start_state,
            horizon=training_config.imagination_horizon,
            discount=training_config.discount,
            lambda_=training_config.lambda_,
            grad_clip_norm=training_config.grad_clip_norm,
            longitudinal_scale=config.env.action.longitudinal_scale,
            lateral_scale=config.env.action.lateral_scale,
            smoothing_factor=config.env.action.smoothing_factor,
            lateral_control=config.env.action.lateral,
            actor_scaler=actor_scaler,
            critic_scaler=critic_scaler,
            amp_context=amp_context,
        )
        behavior_metrics_list.append(behavior_metrics)

    world_model_metrics = _average_metric_dicts(world_model_metrics_list)
    behavior_metrics = _average_metric_dicts(behavior_metrics_list)

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
        evaluation_metrics={},
    )