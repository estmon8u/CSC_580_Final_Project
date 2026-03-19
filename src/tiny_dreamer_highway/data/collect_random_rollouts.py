"""Warm-start replay collection from random actions.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import numpy as np

from tiny_dreamer_highway.config import EnvConfig
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.envs.highway_factory import make_highway_env, make_vectorized_highway_env
from tiny_dreamer_highway.types import Transition
from tiny_dreamer_highway.utils import stabilize_action_array


def _one_hot(index: int, num_actions: int) -> np.ndarray:
    """Create a one-hot float32 vector from an integer action index."""
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[index] = 1.0
    return vec


def collect_random_transitions(
    config: EnvConfig,
    replay_buffer: ReplayBuffer,
    steps: int,
    seed: int | None = None,
) -> int:
    if config.num_envs > 1:
        return _collect_random_transitions_vectorized(config, replay_buffer, steps, seed)
    return _collect_random_transitions_single(config, replay_buffer, steps, seed)


def _collect_random_transitions_single(
    config: EnvConfig,
    replay_buffer: ReplayBuffer,
    steps: int,
    seed: int | None = None,
) -> int:
    """Original single-env random collection path."""
    is_discrete = config.action.is_discrete
    env = make_highway_env(config)
    if seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    added = 0
    previous_action: np.ndarray | None = None

    try:
        for _ in range(steps):
            raw_action = env.action_space.sample()

            if is_discrete:
                env_action = int(raw_action)
                stored_action = _one_hot(env_action, config.action.num_actions)
            else:
                raw_action = np.asarray(raw_action, dtype=np.float32)
                action = stabilize_action_array(
                    raw_action,
                    previous_action=previous_action,
                    longitudinal_scale=config.action.longitudinal_scale,
                    lateral_scale=config.action.lateral_scale,
                    smoothing_factor=config.action.smoothing_factor,
                    lateral_enabled=config.action.lateral,
                )
                previous_action = action
                env_action = action
                stored_action = action

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
                observation, _ = env.reset()
                previous_action = None
    finally:
        env.close()

    return added


def _collect_random_transitions_vectorized(
    config: EnvConfig,
    replay_buffer: ReplayBuffer,
    steps: int,
    seed: int | None = None,
) -> int:
    """Vectorized random collection via SyncVectorEnv."""
    import math

    num_envs = config.num_envs
    is_discrete = config.action.is_discrete
    vec_env = make_vectorized_highway_env(config)
    observations, _ = vec_env.reset(seed=seed)
    # Per-env previous actions for smoothing (continuous only)
    previous_actions: list[np.ndarray | None] = [None] * num_envs

    iterations = math.ceil(steps / num_envs)

    # Accumulate raw batch arrays per step.  After collection we
    # stack along axis=1 to get (N, T, ...) and flush each env's
    # trajectory contiguously via add_batch, avoiding both the
    # interleaved-layout bug and Python-loop insertion overhead.
    hist_obs: list[np.ndarray] = []
    hist_act: list[np.ndarray] = []
    hist_rew: list[np.ndarray] = []
    hist_next_obs: list[np.ndarray] = []
    hist_done: list[np.ndarray] = []
    hist_term: list[np.ndarray] = []
    hist_trunc: list[np.ndarray] = []

    try:
        for _ in range(iterations):
            # SyncVectorEnv action_space.sample() returns (N, action_dim)
            raw_actions = vec_env.action_space.sample()

            if is_discrete:
                env_actions = np.asarray(raw_actions, dtype=np.int64)
                stored_actions = np.stack(
                    [_one_hot(int(a), config.action.num_actions) for a in env_actions]
                )
            else:
                stored_list = []
                for i in range(num_envs):
                    raw = np.asarray(raw_actions[i], dtype=np.float32)
                    action = stabilize_action_array(
                        raw,
                        previous_action=previous_actions[i],
                        longitudinal_scale=config.action.longitudinal_scale,
                        lateral_scale=config.action.lateral_scale,
                        smoothing_factor=config.action.smoothing_factor,
                        lateral_enabled=config.action.lateral,
                    )
                    previous_actions[i] = action
                    stored_list.append(action)
                stored_actions = np.stack(stored_list)
                env_actions = stored_actions

            next_observations, rewards, terminations, truncations, infos = vec_env.step(env_actions)
            dones = terminations | truncations

            # Intercept Gymnasium's auto-reset to get the true terminal frames
            real_next_obs = np.copy(next_observations)
            if "_final_observation" in infos and "final_observation" in infos:
                final_mask = infos["_final_observation"]
                final_frames = infos["final_observation"]
                for i in range(num_envs):
                    if final_mask[i] and final_frames[i] is not None:
                        real_next_obs[i] = final_frames[i]

            # Copy observations because SyncVectorEnv may reuse buffers
            hist_obs.append(np.copy(observations))
            hist_act.append(stored_actions.copy())
            hist_rew.append(rewards.astype(np.float32))
            hist_next_obs.append(real_next_obs)  # already a copy
            hist_done.append(dones.copy())
            hist_term.append(terminations.copy())
            hist_trunc.append(truncations.copy())

            observations = next_observations
            # Reset smoothing for done envs
            for i in range(num_envs):
                if dones[i]:
                    previous_actions[i] = None
    finally:
        vec_env.close()

    # Stack & Transpose: (T, N, ...) → (N, T, ...) then flush per-env
    env_obs = np.stack(hist_obs, axis=1)          # (N, T, ...)
    env_act = np.stack(hist_act, axis=1)          # (N, T, action_dim)
    env_rew = np.stack(hist_rew, axis=1)          # (N, T)
    env_next_obs = np.stack(hist_next_obs, axis=1)
    env_done = np.stack(hist_done, axis=1)        # (N, T)
    env_term = np.stack(hist_term, axis=1)
    env_trunc = np.stack(hist_trunc, axis=1)

    for i in range(num_envs):
        replay_buffer.add_batch(
            observations=env_obs[i],
            actions=env_act[i],
            rewards=env_rew[i],
            next_observations=env_next_obs[i],
            dones=env_done[i],
            terminated=env_term[i],
            truncated=env_trunc[i],
        )

    return num_envs * iterations
