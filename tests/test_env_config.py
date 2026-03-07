import gymnasium as gym
import numpy as np

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
    assert kwargs["duration"] == config.max_episode_steps
    assert kwargs["collision_reward"] == config.reward.collision_reward
    assert kwargs["offroad_terminal"] == config.reward.offroad_terminal


def test_reward_wrapper_penalizes_unstable_steering_and_offroad() -> None:
    config = EnvConfig(
        reward={
            "offroad_penalty": 3.0,
            "steering_penalty": 0.2,
            "steering_change_penalty": 0.3,
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
    config = EnvConfig(reward={"offroad_penalty": 0.0, "steering_penalty": 0.0, "steering_change_penalty": 1.0})
    env = DrivingPenaltyRewardWrapper(DummyEnv(on_road=True), config)
    env.reset()

    _, first_reward, _, _, _ = env.step(np.asarray([0.0, 0.5], dtype=np.float32))
    env.reset()
    _, second_reward, _, _, _ = env.step(np.asarray([0.0, 0.5], dtype=np.float32))

    assert first_reward == second_reward == 0.5
