from tiny_dreamer_highway.config import EnvConfig
from tiny_dreamer_highway.envs.highway_factory import build_highway_env_kwargs


def test_build_highway_env_kwargs_matches_expected_contract() -> None:
    config = EnvConfig(observation_height=64, observation_width=64, frame_stack=1)
    kwargs = build_highway_env_kwargs(config)
    assert kwargs["observation"]["type"] == "GrayscaleObservation"
    assert kwargs["observation"]["observation_shape"] == (64, 64)
    assert kwargs["action"]["type"] == "ContinuousAction"
    assert kwargs["duration"] == config.max_episode_steps
