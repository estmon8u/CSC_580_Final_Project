import json
from pathlib import Path

from tiny_dreamer_highway.evaluation.training_analysis import (
    export_training_history_artifacts,
    load_cycle_metrics_history,
    plot_training_history,
    summarize_training_history,
)


def write_metrics_csv(path: Path) -> Path:
    path.write_text(
        "step,warm_start_added,policy_added,replay_size,world_model/reconstruction_loss,world_model/reconstruction_mse,world_model/reward_loss,world_model/continue_loss,world_model/total_loss,behavior/actor_loss,behavior/critic_loss,behavior/imagined_reward_mean,behavior/imagined_value_mean,evaluation/mean_reward,evaluation/crash_rate\n"
        "1,64,8,72,0.20,0.010,0.10,0.08,0.30,-0.05,0.20,0.01,0.05,1.5,0.60\n"
        "2,0,8,80,0.15,0.008,0.08,0.06,0.23,-0.10,0.18,0.03,0.07,2.0,0.35\n"
        "3,0,8,88,0.12,0.006,0.05,0.04,0.17,-0.12,0.15,0.04,0.09,3.2,0.20\n",
        encoding="utf-8",
    )
    return path


def test_load_cycle_metrics_history_parses_numbers(tmp_path: Path) -> None:
    csv_path = write_metrics_csv(tmp_path / "cycle_metrics.csv")
    history = load_cycle_metrics_history(csv_path)

    assert len(history) == 3
    assert history[0]["step"] == 1
    assert history[1]["replay_size"] == 80
    assert history[2]["world_model/total_loss"] == 0.17


def test_summarize_training_history_finds_best_steps(tmp_path: Path) -> None:
    csv_path = write_metrics_csv(tmp_path / "cycle_metrics.csv")
    history = load_cycle_metrics_history(csv_path)
    summary = summarize_training_history(history)

    assert summary["num_records"] == 3
    assert summary["last_step"] == 3
    assert summary["best_world_model_step"] == 3
    assert summary["best_imagined_reward_step"] == 3
    assert summary["best_eval_reward_step"] == 3
    assert summary["best_eval_mean_reward"] == 3.2
    assert summary["largest_replay_size"] == 88


def test_plot_training_history_writes_png(tmp_path: Path) -> None:
    csv_path = write_metrics_csv(tmp_path / "cycle_metrics.csv")
    history = load_cycle_metrics_history(csv_path)
    plot_path = plot_training_history(history, tmp_path / "training_curves.png")

    assert plot_path.exists()
    assert plot_path.suffix == ".png"


def test_export_training_history_artifacts_writes_summary_and_plot(tmp_path: Path) -> None:
    csv_path = write_metrics_csv(tmp_path / "cycle_metrics.csv")
    outputs = export_training_history_artifacts(csv_path, tmp_path / "analysis", prefix="baseline")

    assert outputs["summary"].exists()
    assert outputs["curves"].exists()

    payload = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert payload["best_world_model_step"] == 3


def test_load_cycle_metrics_history_ignores_overflow_columns_from_old_csv_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "cycle_metrics.csv"
    csv_path.write_text(
        "step,world_model/total_loss\n"
        "1,0.50\n"
        "2,0.25,3.5\n",
        encoding="utf-8",
    )

    history = load_cycle_metrics_history(csv_path)

    assert history == [
        {"step": 1, "world_model/total_loss": 0.5},
        {"step": 2, "world_model/total_loss": 0.25},
    ]