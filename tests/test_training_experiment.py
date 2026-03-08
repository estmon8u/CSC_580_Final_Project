from pathlib import Path

import numpy as np

from tiny_dreamer_highway.config import ExperimentConfig
from tiny_dreamer_highway.training.experiment import (
    infer_env_shapes,
    resolve_training_device,
    run_training_experiment,
)
from tiny_dreamer_highway.training.pipeline import PipelineCycleMetrics
from tiny_dreamer_highway.types import Transition


class _FakeActionSpace:
    def sample(self) -> np.ndarray:
        return np.asarray([0.0, 0.0], dtype=np.float32)


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = _FakeActionSpace()

    def reset(self, seed: int | None = None):
        return np.zeros((1, 64, 64), dtype=np.uint8), {"seed": seed}

    def close(self) -> None:
        return None


def test_resolve_training_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_training_device("cuda").type == "cpu"


def test_infer_env_shapes_uses_env_reset_and_action_sample(monkeypatch) -> None:
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.make_highway_env",
        lambda config: _FakeEnv(),
    )
    config = ExperimentConfig()
    observation_shape, action_dim = infer_env_shapes(config)

    assert observation_shape == (1, 64, 64)
    assert action_dim == 2


def test_run_training_experiment_writes_logs_and_checkpoints(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.make_highway_env",
        lambda config: _FakeEnv(),
    )

    def fake_run_training_cycle(
        config,
        replay_buffer,
        world_model,
        actor,
        critic,
        world_model_optimizer,
        actor_optimizer,
        critic_optimizer,
        warm_start_steps: int = 0,
        policy_steps: int = 0,
        seed: int | None = None,
        *,
        wm_scaler=None,
        actor_scaler=None,
        critic_scaler=None,
        amp_context=None,
    ) -> PipelineCycleMetrics:
        for _ in range(max(1, warm_start_steps + policy_steps)):
            replay_buffer.add(
                Transition(
                    observation=np.zeros((1, 64, 64), dtype=np.uint8),
                    action=np.zeros((2,), dtype=np.float32),
                    reward=0.0,
                    next_observation=np.zeros((1, 64, 64), dtype=np.uint8),
                    done=False,
                )
            )
        return PipelineCycleMetrics(
            warm_start_added=warm_start_steps,
            policy_added=policy_steps,
            replay_size=len(replay_buffer),
            world_model_metrics={
                "reconstruction_loss": 1.0,
                "reward_loss": 0.5,
                "continue_loss": 0.1,
                "total_loss": 1.5,
            },
            behavior_metrics={
                "actor_loss": -0.1,
                "critic_loss": 0.2,
                "imagined_reward_mean": 0.3,
                "imagined_value_mean": 0.4,
            },
            evaluation_metrics={},
        )

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.run_training_cycle",
        fake_run_training_cycle,
    )

    config = ExperimentConfig.model_validate(
        {
            "seed": 7,
            "device": "cpu",
            "training": {
                "batch_size": 4,
                "imagination_horizon": 5,
                "world_model_lr": 3e-4,
                "actor_lr": 8e-5,
                "critic_lr": 8e-5,
                "cycles": 3,
                "warm_start_steps": 4,
                "policy_steps": 2,
                "checkpoint_interval": 2,
            },
        }
    )

    summary = run_training_experiment(config, tmp_path)

    assert summary.completed_cycles == 3
    assert summary.latest_checkpoint is not None
    assert summary.latest_checkpoint.exists()
    assert (tmp_path / "logs" / "cycle_metrics.jsonl").exists()
    assert (tmp_path / "logs" / "cycle_metrics.csv").exists()
    assert (tmp_path / "logs" / "latest_summary.json").exists()


def test_run_training_experiment_overwrites_previous_artifacts_for_fresh_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.make_highway_env",
        lambda config: _FakeEnv(),
    )

    def fake_run_training_cycle(
        config,
        replay_buffer,
        world_model,
        actor,
        critic,
        world_model_optimizer,
        actor_optimizer,
        critic_optimizer,
        warm_start_steps: int = 0,
        policy_steps: int = 0,
        seed: int | None = None,
        *,
        wm_scaler=None,
        actor_scaler=None,
        critic_scaler=None,
        amp_context=None,
    ) -> PipelineCycleMetrics:
        replay_buffer.add(
            Transition(
                observation=np.zeros((1, 64, 64), dtype=np.uint8),
                action=np.zeros((2,), dtype=np.float32),
                reward=0.0,
                next_observation=np.zeros((1, 64, 64), dtype=np.uint8),
                done=False,
            )
        )
        return PipelineCycleMetrics(
            warm_start_added=warm_start_steps,
            policy_added=policy_steps,
            replay_size=len(replay_buffer),
            world_model_metrics={"total_loss": 1.5},
            behavior_metrics={"actor_loss": -0.1, "critic_loss": 0.2},
            evaluation_metrics={},
        )

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.run_training_cycle",
        fake_run_training_cycle,
    )

    stale_checkpoint = tmp_path / "checkpoints" / "checkpoint_99999.pt"
    stale_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stale_checkpoint.write_text("stale", encoding="utf-8")
    stale_csv = tmp_path / "logs" / "cycle_metrics.csv"
    stale_csv.parent.mkdir(parents=True, exist_ok=True)
    stale_csv.write_text("old,data\n", encoding="utf-8")

    config = ExperimentConfig.model_validate(
        {
            "seed": 7,
            "device": "cpu",
            "training": {
                "batch_size": 4,
                "imagination_horizon": 5,
                "world_model_lr": 3e-4,
                "actor_lr": 8e-5,
                "critic_lr": 8e-5,
                "cycles": 1,
                "warm_start_steps": 0,
                "policy_steps": 0,
                "checkpoint_interval": 1,
            },
        }
    )

    summary = run_training_experiment(config, tmp_path)

    assert summary.latest_checkpoint is not None
    assert summary.latest_checkpoint.exists()
    assert not stale_checkpoint.exists()
    csv_text = (tmp_path / "logs" / "cycle_metrics.csv").read_text(encoding="utf-8")
    assert "old,data" not in csv_text