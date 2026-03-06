"""Environment factory helpers for Highway-Env.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from typing import Any

from tiny_dreamer_highway.config import EnvConfig


def build_highway_env_kwargs(config: EnvConfig) -> dict[str, Any]:
    return {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (config.observation_height, config.observation_width),
            "stack_size": config.frame_stack,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": config.action.longitudinal,
            "lateral": config.action.lateral,
        },
        "duration": config.max_episode_steps,
    }


def make_highway_env(config: EnvConfig):
    import gymnasium as gym
    import highway_env  # noqa: F401

    env = gym.make(config.env_id, render_mode="rgb_array")
    env.unwrapped.configure(build_highway_env_kwargs(config))
    return env
