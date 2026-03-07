import json
from pathlib import Path

from tiny_dreamer_highway.training import (
    PipelineCycleMetrics,
    append_metrics_csv,
    append_metrics_jsonl,
    export_cycle_metrics,
    flatten_cycle_metrics,
    write_artifact_summary,
)


def make_metrics() -> PipelineCycleMetrics:
    return PipelineCycleMetrics(
        warm_start_added=16,
        policy_added=4,
        replay_size=20,
        world_model_metrics={
            "reconstruction_loss": 0.1,
            "reward_loss": 0.2,
            "total_loss": 0.3,
        },
        behavior_metrics={
            "actor_loss": 0.4,
            "critic_loss": 0.5,
            "imagined_reward_mean": 0.6,
            "imagined_value_mean": 0.7,
        },
        evaluation_metrics={
            "mean_reward": 1.2,
            "mean_steps": 33.0,
        },
    )


def test_flatten_cycle_metrics_returns_prefixed_record() -> None:
    record = flatten_cycle_metrics(3, make_metrics())
    assert record["step"] == 3
    assert record["warm_start_added"] == 16
    assert record["world_model/total_loss"] == 0.3
    assert record["behavior/critic_loss"] == 0.5
    assert record["evaluation/mean_reward"] == 1.2


def test_append_metrics_jsonl_and_csv_write_records(tmp_path: Path) -> None:
    record = flatten_cycle_metrics(1, make_metrics())
    jsonl_path = append_metrics_jsonl(tmp_path / "cycle_metrics.jsonl", record)
    csv_path = append_metrics_csv(tmp_path / "cycle_metrics.csv", record)

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["step"] == 1

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "world_model/total_loss" in csv_text
    assert "behavior/actor_loss" in csv_text


def test_write_artifact_summary_persists_latest_checkpoint_and_metrics(tmp_path: Path) -> None:
    record = flatten_cycle_metrics(2, make_metrics())
    summary_path = write_artifact_summary(
        tmp_path / "latest_summary.json",
        step=2,
        record=record,
        checkpoint_file=tmp_path / "checkpoint_00002.pt",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["latest_step"] == 2
    assert summary["latest_metrics"]["behavior/imagined_value_mean"] == 0.7
    assert summary["latest_metrics"]["evaluation/mean_reward"] == 1.2
    assert summary["checkpoint_file"].endswith("checkpoint_00002.pt")


def test_export_cycle_metrics_writes_all_artifacts(tmp_path: Path) -> None:
    outputs = export_cycle_metrics(
        tmp_path,
        step=5,
        metrics=make_metrics(),
        checkpoint_file=tmp_path / "checkpoint_00005.pt",
    )

    assert set(outputs.keys()) == {"jsonl", "csv", "summary"}
    assert outputs["jsonl"].exists()
    assert outputs["csv"].exists()
    assert outputs["summary"].exists()