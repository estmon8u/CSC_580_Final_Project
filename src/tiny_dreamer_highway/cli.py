"""Command-line entry points for early project workflows.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_dreamer_highway.config import ExperimentConfig, load_experiment_config


def summarize_config(config: ExperimentConfig) -> str:
    return (
        f"Loaded config for {config.env.env_id} | "
        f"obs={config.env.observation_height}x{config.env.observation_width} | "
        f"replay_capacity={config.replay.capacity} | "
        f"sequence_length={config.replay.sequence_length} | "
        f"batch_size={config.training.batch_size}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny Dreamer Highway CLI")
    subparsers = parser.add_subparsers(dest="command")

    show_config = subparsers.add_parser(
        "show-config",
        help="Load an experiment config and print a compact summary.",
    )
    show_config.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML file.",
    )

    parser.set_defaults(command="show-config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    print(summarize_config(config))


if __name__ == "__main__":
    main()
