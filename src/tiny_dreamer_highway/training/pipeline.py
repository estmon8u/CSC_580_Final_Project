"""Alternating training pipeline for the Tiny Dreamer Highway agent.

This module implements the DreamerV1 training loop at the *cycle* level.
Each training cycle performs these steps in sequence:

1. **Warm-start collection** (cycle 1 only) — fill the replay buffer
   with random-policy transitions so the world model has enough data
   to begin learning.
2. **World-model updates** — sample sequence batches from the replay
   buffer and train the world model (encoder → RSSM → decoder + heads).
3. **Behavior updates** — use posterior states from the world-model
   updates as starting points, imagine future trajectories through
   the learned dynamics, and train the actor-critic via
   λ-returns.
4. **Policy collection** — interact with the real environment using
   the learned actor to gather fresh transitions for the replay buffer.

The outer experiment runner (``experiment.py``) calls
``run_training_cycle()`` in a loop and handles checkpointing,
evaluation, and logging.

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
from tiny_dreamer_highway.models.discrete_actor import DiscreteActor
from tiny_dreamer_highway.training.behavior_learning import train_behavior_step
from tiny_dreamer_highway.training.sequence_world_model_step import (
    train_sequence_world_model_step,
)
from tiny_dreamer_highway.training.world_model_step import train_world_model_step
from tiny_dreamer_highway.types import Transition
from tiny_dreamer_highway.utils import stabilize_action_tensor


@dataclass(slots=True)
class PipelineCycleMetrics:
    """Metrics returned by a single training cycle.

    Aggregates counts, replay statistics, and averaged loss dicts from
    the world-model and behavior update loops.
    """

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


def _observation_to_tensor(observation: np.ndarray, device: torch.device | None = None) -> Tensor:
    """Convert a single observation array to a batched 4-D tensor.

    Highway-env returns observations as ``(H, W)`` or ``(C, H, W)``
    arrays.  This function adds the necessary leading dimensions so
    that the tensor is always ``(1, C, H, W)`` for the world model.
    """
    observation_tensor = torch.as_tensor(observation, device=device)
    if observation_tensor.ndim == 2:
        observation_tensor = observation_tensor.unsqueeze(0)  # add channel dim
    if observation_tensor.ndim == 3:
        observation_tensor = observation_tensor.unsqueeze(0)  # add batch dim
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
    """Top up the replay buffer until it supports sequence sampling.

    After the initial warm-start, the replay buffer may still lack
    enough *contiguous, non-terminal* runs of length ``sequence_length``
    to form a full batch (e.g., if early episodes terminated quickly).
    This function adds random-policy transitions in chunks until
    ``can_sample()`` returns True.

    Returns:
        Number of extra transitions added (0 if the buffer was already
        sampleable).

    Raises:
        ValueError: If ``sequence_length > max_episode_steps`` (sampling
            is structurally impossible).
    """
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
    """Compute a posterior latent state from observations and actions.

    Used as a fallback to initialize behavior-learning start states when
    no posterior states were collected during the world-model update loop.
    Runs a single forward pass through the world model with gradients
    disabled.
    """
    ctx = amp_context if amp_context is not None else nullcontext()
    with torch.no_grad(), ctx:
        output = world_model(observations, actions)
    return output.posterior_state


def collect_actor_transitions(
    config: ExperimentConfig,
    replay_buffer: ReplayBuffer,
    world_model: TinyWorldModel,
    actor: Actor | DiscreteActor,
    steps: int,
    seed: int | None = None,
) -> int:
    """Collect transitions using the learned policy in a single environment.

    At each step:
    1. Encode the current observation through the world model.
    2. Update the recurrent latent state (posterior).
    3. Sample an action from the actor conditioned on the latent features.
    4. Step the environment and store the transition in the replay buffer.
    5. On episode termination, reset the environment and latent state.

    Args:
        config:        Full experiment config (env and action settings).
        replay_buffer: Target replay buffer to store transitions.
        world_model:   Trained world model for observation encoding.
        actor:         Trained actor (continuous or discrete).
        steps:         Number of environment steps to collect.
        seed:          Optional seed for environment and action space.

    Returns:
        Number of transitions successfully added to the replay buffer.
    """
    if steps <= 0:
        return 0
    is_discrete = config.env.action.is_discrete
    env = make_highway_env(config.env)
    if seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    model_device = _module_device(world_model)
    action_dim = world_model.rssm.action_dim
    # Initialize latent state and action to zeros for the first step
    prev_state = world_model.rssm.initial_state(batch_size=1, device=model_device)
    prev_action = torch.zeros(1, action_dim, device=model_device)
    added = 0

    try:
        for _ in range(steps):
            with torch.inference_mode():
                # Encode the current frame and update the recurrent latent state
                observation_tensor = _observation_to_tensor(
                    np.asarray(observation), device=model_device
                )
                posterior = world_model(observation_tensor, prev_action, prev_state=prev_state)
                prev_state = posterior.posterior_state

                # Sample action from the actor using the latent features
                raw_action_tensor = actor(prev_state.features)
                if is_discrete:
                    action_tensor = raw_action_tensor
                    # Store one-hot action in replay; convert to int for env
                    stored_action = action_tensor.squeeze(0).float().cpu().numpy()
                    env_action = int(raw_action_tensor.squeeze(0).argmax(dim=-1).item())
                else:
                    # Apply action stabilization (smoothing, scaling) before
                    # sending to the environment
                    action_tensor = stabilize_action_tensor(
                        raw_action_tensor,
                        previous_action=prev_action,
                        longitudinal_scale=config.env.action.longitudinal_scale,
                        lateral_scale=config.env.action.lateral_scale,
                        smoothing_factor=config.env.action.smoothing_factor,
                        lateral_enabled=config.env.action.lateral,
                    )
                    stored_action = action_tensor.squeeze(0).float().cpu().numpy()
                    env_action = stored_action
                prev_action = action_tensor

            next_observation, reward, terminated, truncated, _ = env.step(env_action)
            done = bool(terminated or truncated)
            replay_buffer.add(
                Transition(
                    observation=np.asarray(observation),
                    action=stored_action,
                    reward=float(reward),
                    next_observation=np.asarray(next_observation),
                    done=done,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            added += 1
            observation = next_observation

            if done:
                # Episode ended — reset environment and latent state
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
    """Execute one full train–collect cycle.

    This is the inner loop of the Dreamer training procedure:

    1. Optionally add random warm-start transitions (first cycle only).
    2. Ensure the replay buffer has enough valid contiguous sequences.
    3. Run ``world_model_updates_per_cycle`` world-model gradient steps,
       collecting posterior states as behavior start-state candidates.
    4. Run ``behavior_updates_per_cycle`` actor-critic gradient steps
       using imagined rollouts from the collected posteriors.
    5. Collect ``policy_steps`` real-environment transitions using the
       updated actor.

    Returns:
        ``PipelineCycleMetrics`` with counts and averaged loss dicts.
    """
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
    # Collect posterior states from every WM training step to use as
    # start states for the behavior (actor-critic) update.
    all_posterior_states: list[LatentState] = []
    for _ in range(training_config.world_model_updates_per_cycle):
        seq_batch = replay_buffer.sample_sequence_batch(
            batch_size=batch_size, sequence_length=sequence_length,
        )
        # IMPORTANT: Replay stores (obs_t, action_t, reward_t, next_obs_t)
        # where action_t leads FROM obs_t TO next_obs_t.  The RSSM
        # observe_step advances the deterministic state with the supplied
        # action *before* conditioning on the observation embedding, so
        # the observation paired with action_t must be the POST-action
        # observation = next_obs_t.
        # non_blocking=True lets PCIe transfer overlap with Python work.
        observations = torch.from_numpy(seq_batch.next_observations).to(device=model_device, non_blocking=True)
        actions = torch.from_numpy(seq_batch.actions).to(dtype=torch.float32, device=model_device, non_blocking=True)
        rewards = torch.from_numpy(seq_batch.rewards).to(dtype=torch.float32, device=model_device, non_blocking=True)

        outputs, world_model_metrics = train_sequence_world_model_step(
            world_model,
            world_model_optimizer,
            observations,
            actions,
            rewards,
            terminals=torch.from_numpy(seq_batch.terminals).to(dtype=torch.float32, device=model_device, non_blocking=True),
            kl_weight=training_config.kl_weight,
            kl_balance=training_config.kl_balance,
            free_nats=training_config.free_nats,
            continue_loss_weight=training_config.continue_loss_weight,
            overshooting_horizon=training_config.overshooting_horizon,
            overshooting_kl_weight=training_config.overshooting_kl_weight,
            grad_clip_norm=training_config.grad_clip_norm,
            grad_scaler=wm_scaler,
            amp_context=amp_context,
        )
        world_model_metrics_list.append(world_model_metrics)
        # Detach and save posterior states from all sequence time steps.
        # These will be used as imagination start states for the actor-
        # critic behavior update.
        for wm_output in outputs:
            all_posterior_states.append(
                LatentState(
                    deterministic=wm_output.posterior_state.deterministic.detach(),
                    stochastic=wm_output.posterior_state.stochastic.detach(),
                )
            )

    # Pre-concatenate all posterior states once so that random indexing
    # in the behavior loop is a single GPU gather rather than repeated
    # list indexing + stacking.
    if all_posterior_states:
        all_det = torch.cat([s.deterministic for s in all_posterior_states], dim=0)
        all_sto = torch.cat([s.stochastic for s in all_posterior_states], dim=0)
        n_total = all_det.shape[0]
    else:
        all_det = all_sto = None
        n_total = 0

    behavior_metrics_list: list[dict[str, float]] = []
    for _ in range(training_config.behavior_updates_per_cycle):
        # Sample random start states from the pre-concatenated posteriors
        if all_det is not None:
            indices = torch.randint(0, n_total, (batch_size,), device=model_device)
            start_state = LatentState(
                deterministic=all_det[indices],
                stochastic=all_sto[indices],
            )
        else:
            # Fallback: seed from replay buffer when no posteriors available
            # (uses the same post-action alignment as the WM training)
            batch = replay_buffer.sample_batch(batch_size=batch_size)
            observations = torch.from_numpy(batch.next_observations).to(device=model_device, non_blocking=True)
            actions = torch.from_numpy(batch.actions).to(dtype=torch.float32, device=model_device, non_blocking=True)
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
            actor_entropy_weight=training_config.actor_entropy_weight,
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