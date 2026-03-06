from pathlib import Path

from tiny_dreamer_highway.config import ExperimentConfig, load_experiment_config


def test_load_experiment_config_reads_example() -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "base_experiment.yaml"
    config = load_experiment_config(config_path)
    assert isinstance(config, ExperimentConfig)
    assert config.env.env_id == "highway-v0"
    assert config.replay.sequence_length == 8
    assert config.training.imagination_horizon == 5
