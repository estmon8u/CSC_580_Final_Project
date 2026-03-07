from pathlib import Path

import numpy as np
import pytest

from tiny_dreamer_highway.cli import (
    build_parser,
    run_collect_random,
    run_train_baseline,
    summarize_collection,
    summarize_config,
    summarize_training_run,
)
from tiny_dreamer_highway.config import load_experiment_config
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.training.experiment import TrainingRunSummary
from tiny_dreamer_highway.types import Transition


def test_parser_defaults_to_show_config() -> None:
    parser = build_parser()
    args = parser.parse_args(["show-config", "--config", "examples/base_experiment.yaml"])
    assert args.command == "show-config"
    assert args.config == Path("examples/base_experiment.yaml")


def test_parser_supports_collect_random() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["collect-random", "--config", "examples/base_experiment.yaml", "--steps", "24"]
    )
    assert args.command == "collect-random"
    assert args.config == Path("examples/base_experiment.yaml")
    assert args.steps == 24


def test_parser_supports_train_baseline() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-baseline",
            "--config",
            "examples/base_experiment.yaml",
            "--artifact-root",
            "artifacts/local_run",
            "--cycles",
            "12",
        ]
    )
    assert args.command == "train-baseline"
    assert args.config == Path("examples/base_experiment.yaml")
    assert args.artifact_root == Path("artifacts/local_run")
    assert args.cycles == 12


def test_summarize_config_contains_expected_fields() -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "base_experiment.yaml"
    config = load_experiment_config(config_path)
    summary = summarize_config(config)
    assert "highway-v0" in summary
    assert "replay_capacity=10000" in summary
    assert "sequence_length=8" in summary
    assert "batch_size=4" in summary


def test_summarize_collection_contains_sampling_shapes() -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "base_experiment.yaml"
    config = load_experiment_config(config_path)
    replay_buffer = ReplayBuffer(capacity=config.replay.capacity)

    for seed in range(config.replay.sequence_length):
        replay_buffer.add(
            Transition(
                observation=np.ones((4, 4), dtype=np.uint8),
                action=np.asarray([seed, seed + 0.5], dtype=np.float32),
                reward=float(seed),
                next_observation=np.full((4, 4), 2, dtype=np.uint8),
                done=False,
            )
        )

    summary = summarize_collection(config, replay_buffer, added=config.replay.sequence_length)
    assert "Collected 8 transitions into replay" in summary
    assert "replay_size=8" in summary
    assert "batch_obs_shape=(4, 4, 4)" in summary
    assert "sequence_batch=4x8" in summary


def test_run_collect_random_uses_collection_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "base_experiment.yaml"

    def fake_collect_random_transitions(
        env_config,
        replay_buffer: ReplayBuffer,
        steps: int,
        seed: int | None = None,
    ) -> int:
        assert seed == 7
        for seed in range(steps):
            replay_buffer.add(
                Transition(
                    observation=np.full((4, 4), seed, dtype=np.uint8),
                    action=np.asarray([seed, seed + 0.5], dtype=np.float32),
                    reward=float(seed),
                    next_observation=np.full((4, 4), seed + 1, dtype=np.uint8),
                    done=False,
                )
            )
        return steps

    monkeypatch.setattr(
        "tiny_dreamer_highway.cli.collect_random_transitions",
        fake_collect_random_transitions,
    )

    summary = run_collect_random(config_path, steps=8)
    assert "Collected 8 transitions into replay" in summary
    assert "replay_size=8" in summary
    assert "sequence_batch=4x8" in summary


def test_summarize_training_run_contains_key_metrics() -> None:
    summary = TrainingRunSummary(
        total_cycles=5,
        completed_cycles=5,
        replay_size=32,
        latest_record={
            "world_model/total_loss": 1.5,
            "behavior/actor_loss": -0.1,
            "behavior/critic_loss": 0.2,
            "evaluation/mean_reward": 7.5,
        },
        latest_checkpoint=Path("artifacts/checkpoints/checkpoint_00005.pt"),
        checkpoint_dir=Path("artifacts/checkpoints"),
        log_dir=Path("artifacts/logs"),
    )

    text = summarize_training_run(summary)
    assert "Completed 5 training cycles" in text
    assert "replay_size=32" in text
    assert "checkpoint_00005.pt" in text


def test_run_train_baseline_uses_training_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "base_experiment.yaml"
    artifact_root = Path("artifacts/local_run")

    def fake_run_training_experiment(
        config,
        artifact_root: Path,
        *,
        cycles: int | None = None,
        warm_start_steps: int | None = None,
        policy_steps: int | None = None,
        checkpoint_interval: int | None = None,
        resume_from: Path | None = None,
    ) -> TrainingRunSummary:
        assert artifact_root == Path("artifacts/local_run")
        assert cycles == 3
        return TrainingRunSummary(
            total_cycles=3,
            completed_cycles=3,
            replay_size=19,
            latest_record={
                "world_model/total_loss": 0.5,
                "behavior/actor_loss": -0.2,
                "behavior/critic_loss": 0.1,
                "evaluation/mean_reward": 4.2,
            },
            latest_checkpoint=Path("artifacts/local_run/checkpoints/checkpoint_00003.pt"),
            checkpoint_dir=Path("artifacts/local_run/checkpoints"),
            log_dir=Path("artifacts/local_run/logs"),
        )

    monkeypatch.setattr(
        "tiny_dreamer_highway.cli.run_training_experiment",
        fake_run_training_experiment,
    )

    summary = run_train_baseline(config_path, artifact_root, cycles=3)
    assert "Completed 3 training cycles" in summary
