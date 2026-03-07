from pathlib import Path

from tiny_dreamer_highway.config import ExperimentConfig, load_experiment_config


def test_load_experiment_config_reads_example() -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "base_experiment.yaml"
    config = load_experiment_config(config_path)
    assert isinstance(config, ExperimentConfig)
    assert config.env.env_id == "highway-v0"
    assert config.env.reward.offroad_penalty == 3.0
    assert config.replay.sequence_length == 8
    assert config.training.imagination_horizon == 5
    assert config.training.overshooting_horizon == 2
    assert config.training.overshooting_kl_weight == 0.5
    assert config.model.reward_distribution_std == 1.0
    assert config.model.critic_distribution_std == 1.0
    assert config.model.observation_distribution_std == 1.0
