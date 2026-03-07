"""Environment factory helpers for Highway-Env.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from tiny_dreamer_highway.config import EnvConfig


class DrivingPenaltyRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: EnvConfig) -> None:
        super().__init__(env)
        self._config = config
        self._previous_lateral_action = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._previous_lateral_action = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward) - self._compute_penalty(action)
        return observation, shaped_reward, terminated, truncated, info

    def _compute_penalty(self, action: Any) -> float:
        penalties = self._config.reward
        lateral_action = _extract_lateral_action(action, self._config)
        penalty = penalties.steering_penalty * abs(lateral_action)
        penalty += penalties.steering_change_penalty * abs(
            lateral_action - self._previous_lateral_action
        )
        self._previous_lateral_action = lateral_action

        vehicle = getattr(self.env.unwrapped, "vehicle", None)
        if vehicle is not None and not bool(getattr(vehicle, "on_road", True)):
            penalty += penalties.offroad_penalty
        return float(penalty)


def _extract_lateral_action(action: Any, config: EnvConfig) -> float:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.size == 0 or not config.action.lateral:
        return 0.0
    if config.action.longitudinal and action_array.size >= 2:
        return float(action_array[1])
    return float(action_array[0])


def _should_apply_reward_wrapper(config: EnvConfig) -> bool:
    reward_config = config.reward
    return (
        reward_config.offroad_penalty > 0.0
        or reward_config.steering_penalty > 0.0
        or reward_config.steering_change_penalty > 0.0
    )


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
        "collision_reward": config.reward.collision_reward,
        "right_lane_reward": config.reward.right_lane_reward,
        "high_speed_reward": config.reward.high_speed_reward,
        "lane_change_reward": config.reward.lane_change_reward,
        "reward_speed_range": list(config.reward.reward_speed_range),
        "normalize_reward": config.reward.normalize_reward,
        "offroad_terminal": config.reward.offroad_terminal,
    }


def make_highway_env(config: EnvConfig):
    import highway_env  # noqa: F401

    env = gym.make(config.env_id, render_mode="rgb_array")
    env.unwrapped.configure(build_highway_env_kwargs(config))
    if _should_apply_reward_wrapper(config):
        env = DrivingPenaltyRewardWrapper(env, config)
    return env
