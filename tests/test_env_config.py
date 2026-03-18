import gymnasium as gym
import numpy as np
import pytest

from tiny_dreamer_highway.config import EnvConfig
from tiny_dreamer_highway.envs.highway_factory import (
    DrivingPenaltyRewardWrapper,
    build_highway_env_kwargs,
)


class DummyVehicle:
    def __init__(self, on_road: bool = True) -> None:
        self.on_road = on_road


class DummyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, on_road: bool = True) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 4), dtype=np.uint8)
        self.vehicle = DummyVehicle(on_road=on_road)

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros((4, 4), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((4, 4), dtype=np.uint8), 1.0, False, False, {"action": action}


def test_build_highway_env_kwargs_matches_expected_contract() -> None:
    config = EnvConfig(observation_height=64, observation_width=64, frame_stack=1)
    kwargs = build_highway_env_kwargs(config)
    assert kwargs["observation"]["type"] == "GrayscaleObservation"
    assert kwargs["observation"]["observation_shape"] == (64, 64)
    assert kwargs["action"]["type"] == "ContinuousAction"
    assert kwargs["lanes_count"] == config.lanes_count
    assert kwargs["vehicles_count"] == config.vehicles_count
    assert kwargs["simulation_frequency"] == config.simulation_frequency
    assert kwargs["policy_frequency"] == config.policy_frequency
    assert kwargs["duration"] == config.max_episode_steps / config.policy_frequency
    assert kwargs["collision_reward"] == config.reward.collision_reward
    assert kwargs["offroad_terminal"] == config.reward.offroad_terminal


def test_build_highway_env_kwargs_discrete_uses_discrete_meta_action() -> None:
    from tiny_dreamer_highway.config import ActionConfig
    config = EnvConfig(
        observation_height=64,
        observation_width=64,
        frame_stack=1,
        action=ActionConfig(type="discrete", num_actions=5),
    )
    kwargs = build_highway_env_kwargs(config)
    assert kwargs["action"]["type"] == "DiscreteMetaAction"
    assert "target_speeds" in kwargs["action"]
    assert len(kwargs["action"]["target_speeds"]) == 5


def test_env_config_rejects_simulation_freq_below_policy_freq() -> None:
    with pytest.raises(ValueError, match="simulation_frequency.*must be >= policy_frequency"):
        EnvConfig(simulation_frequency=3, policy_frequency=10)


def test_reward_wrapper_penalizes_unstable_steering_and_offroad() -> None:
    config = EnvConfig(
        reward={
            "offroad_penalty": 3.0,
            "steering_penalty": 0.2,
            "steering_change_penalty": 0.3,
            "normalize_reward": False,
        }
    )
    env = DrivingPenaltyRewardWrapper(DummyEnv(on_road=False), config)
    env.reset()

    _, reward, _, _, _ = env.step(np.asarray([0.1, 0.5], dtype=np.float32))

    # base reward = 1.0
    # steering_penalty: 0.2 * |0.5| = 0.1
    # steering_change_penalty: 0.3 * |0.5 - 0.0| = 0.15
    # offroad_penalty: 3.0
    # total penalty = 3.25  =>  shaped = 1.0 - 3.25 = -2.25
    assert reward == -2.25


def test_reward_wrapper_resets_steering_history() -> None:
    config = EnvConfig(reward={"offroad_penalty": 0.0, "steering_penalty": 0.0, "steering_change_penalty": 1.0, "normalize_reward": False})
    env = DrivingPenaltyRewardWrapper(DummyEnv(on_road=True), config)
    env.reset()

    _, first_reward, _, _, _ = env.step(np.asarray([0.0, 0.5], dtype=np.float32))
    env.reset()
    _, second_reward, _, _, _ = env.step(np.asarray([0.0, 0.5], dtype=np.float32))

    assert first_reward == second_reward == 0.5
