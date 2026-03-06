"""Command-line entry points for early project workflows.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_dreamer_highway.config import load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny Dreamer Highway CLI")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    print(
        "Loaded config for "
        f"{config.env.env_id} with replay capacity {config.replay.capacity} "
        f"and batch size {config.training.batch_size}."
    )


if __name__ == "__main__":
    main()
