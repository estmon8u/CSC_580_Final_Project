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
from tiny_dreamer_highway.training.experiment import run_training_experiment


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


def summarize_training_run(summary) -> str:
    checkpoint_text = str(summary.latest_checkpoint) if summary.latest_checkpoint is not None else "none"
    return (
        f"Completed {summary.completed_cycles} training cycles | "
        f"replay_size={summary.replay_size} | "
        f"latest_checkpoint={checkpoint_text} | "
        f"world_total_loss={summary.latest_record.get('world_model/total_loss')} | "
        f"actor_loss={summary.latest_record.get('behavior/actor_loss')} | "
        f"critic_loss={summary.latest_record.get('behavior/critic_loss')}"
    )


def run_train_baseline(
    config_path: Path,
    artifact_root: Path,
    *,
    cycles: int | None = None,
    warm_start_steps: int | None = None,
    policy_steps: int | None = None,
    checkpoint_interval: int | None = None,
    resume_from: Path | None = None,
) -> str:
    config = load_experiment_config(config_path)
    summary = run_training_experiment(
        config,
        artifact_root,
        cycles=cycles,
        warm_start_steps=warm_start_steps,
        policy_steps=policy_steps,
        checkpoint_interval=checkpoint_interval,
        resume_from=resume_from,
    )
    return summarize_training_run(summary)


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

    train_baseline = subparsers.add_parser(
        "train-baseline",
        help="Run a real multi-cycle baseline training job and save logs/checkpoints.",
    )
    train_baseline.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML file.",
    )
    train_baseline.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory where checkpoints and logs will be written.",
    )
    train_baseline.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Optional override for the number of training cycles.",
    )
    train_baseline.add_argument(
        "--warm-start-steps",
        type=int,
        default=None,
        help="Optional override for the first-cycle random warm-start steps.",
    )
    train_baseline.add_argument(
        "--policy-steps",
        type=int,
        default=None,
        help="Optional override for actor-driven collection steps per cycle.",
    )
    train_baseline.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Optional override for checkpoint save frequency in cycles.",
    )
    train_baseline.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )

    parser.set_defaults(command="show-config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "collect-random":
        print(run_collect_random(args.config, steps=args.steps))
        return
    if args.command == "train-baseline":
        print(
            run_train_baseline(
                args.config,
                args.artifact_root,
                cycles=args.cycles,
                warm_start_steps=args.warm_start_steps,
                policy_steps=args.policy_steps,
                checkpoint_interval=args.checkpoint_interval,
                resume_from=args.resume_from,
            )
        )
        return

    print(run_show_config(args.config))


if __name__ == "__main__":
    main()
