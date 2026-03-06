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
from tiny_dreamer_highway.data.collect_random_rollouts import collect_random_transitions
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer


def summarize_config(config: ExperimentConfig) -> str:
    return (
        f"Loaded config for {config.env.env_id} | "
        f"obs={config.env.observation_height}x{config.env.observation_width} | "
        f"replay_capacity={config.replay.capacity} | "
        f"sequence_length={config.replay.sequence_length} | "
        f"batch_size={config.training.batch_size}"
    )


def summarize_collection(config: ExperimentConfig, replay_buffer: ReplayBuffer, added: int) -> str:
    batch = replay_buffer.sample_batch(batch_size=config.replay.batch_size)
    sequences = replay_buffer.sample_sequences(
        batch_size=config.replay.batch_size,
        sequence_length=config.replay.sequence_length,
    )
    return (
        f"Collected {added} transitions into replay | "
        f"replay_size={len(replay_buffer)} | "
        f"batch_obs_shape={tuple(batch.observations.shape)} | "
        f"batch_action_shape={tuple(batch.actions.shape)} | "
        f"sequence_batch={len(sequences)}x{len(sequences[0])}"
    )


def run_show_config(config_path: Path) -> str:
    config = load_experiment_config(config_path)
    return summarize_config(config)


def run_collect_random(config_path: Path, steps: int) -> str:
    config = load_experiment_config(config_path)
    replay_buffer = ReplayBuffer(capacity=config.replay.capacity)
    added = collect_random_transitions(config.env, replay_buffer, steps=steps, seed=config.seed)
    return summarize_collection(config, replay_buffer, added)


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

    collect_random = subparsers.add_parser(
        "collect-random",
        help="Collect random rollouts and verify replay sampling.",
    )
    collect_random.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML file.",
    )
    collect_random.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of random environment steps to collect.",
    )

    parser.set_defaults(command="show-config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "collect-random":
        print(run_collect_random(args.config, steps=args.steps))
        return

    print(run_show_config(args.config))


if __name__ == "__main__":
    main()
