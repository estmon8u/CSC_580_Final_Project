from pathlib import Path

from tiny_dreamer_highway.cli import build_parser, summarize_config
from tiny_dreamer_highway.config import load_experiment_config


def test_parser_defaults_to_show_config() -> None:
    parser = build_parser()
    args = parser.parse_args(["show-config", "--config", "examples/base_experiment.yaml"])
    assert args.command == "show-config"
    assert args.config == Path("examples/base_experiment.yaml")


def test_summarize_config_contains_expected_fields() -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "base_experiment.yaml"
    config = load_experiment_config(config_path)
    summary = summarize_config(config)
    assert "highway-v0" in summary
    assert "replay_capacity=10000" in summary
    assert "sequence_length=8" in summary
    assert "batch_size=4" in summary
