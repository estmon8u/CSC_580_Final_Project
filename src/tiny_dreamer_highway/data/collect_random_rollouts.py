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
from tiny_dreamer_highway.envs.highway_factory import make_highway_env
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
                # raw_action is an integer; store one-hot for RSSM compatibility
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
                    observation=np.asarray(observation, dtype=np.uint8),
                    action=stored_action,
                    reward=float(reward),
                    next_observation=np.asarray(next_observation, dtype=np.uint8),
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
