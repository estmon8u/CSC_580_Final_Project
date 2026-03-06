"""Warm-start replay collection from random actions.

Name: Esteban
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


def collect_random_transitions(
    config: EnvConfig,
    replay_buffer: ReplayBuffer,
    steps: int,
    seed: int | None = None,
) -> int:
    env = make_highway_env(config)
    if seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    added = 0

    try:
        for _ in range(steps):
            action = np.asarray(env.action_space.sample(), dtype=np.float32)
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
    finally:
        env.close()

    return added
