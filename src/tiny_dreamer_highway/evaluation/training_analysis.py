"""Training history analysis helpers for Tiny Dreamer Highway.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


INT_FIELDS = {"step", "warm_start_added", "policy_added", "replay_size"}


def _parse_metric_value(key: str, value: str) -> int | float:
    if key in INT_FIELDS:
        return int(float(value))
    return float(value)


def load_cycle_metrics_history(metrics_csv: str | Path) -> list[dict[str, int | float]]:
    path = Path(metrics_csv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        history = [
            {key: _parse_metric_value(key, value) for key, value in row.items() if value not in (None, "")}
            for row in reader
        ]
    return history


def summarize_training_history(history: list[dict[str, int | float]]) -> dict[str, Any]:
    if not history:
        raise ValueError("history must contain at least one record")

    last = history[-1]
    best_world = min(history, key=lambda row: float(row["world_model/total_loss"]))
    best_reward = max(history, key=lambda row: float(row["behavior/imagined_reward_mean"]))
    largest_replay = max(history, key=lambda row: int(row["replay_size"]))
    return {
        "num_records": len(history),
        "last_step": int(last["step"]),
        "last_metrics": dict(last),
        "best_world_model_step": int(best_world["step"]),
        "best_world_model_total_loss": float(best_world["world_model/total_loss"]),
        "best_imagined_reward_step": int(best_reward["step"]),
        "best_imagined_reward_mean": float(best_reward["behavior/imagined_reward_mean"]),
        "largest_replay_step": int(largest_replay["step"]),
        "largest_replay_size": int(largest_replay["replay_size"]),
    }


def plot_training_history(
    history: list[dict[str, int | float]],
    output_path: str | Path,
    *,
    title: str = "Tiny Dreamer Training History",
) -> Path:
    if not history:
        raise ValueError("history must contain at least one record")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(row["step"]) for row in history]
    world_total = [float(row["world_model/total_loss"]) for row in history]
    world_recon = [float(row["world_model/reconstruction_loss"]) for row in history]
    world_reward = [float(row["world_model/reward_loss"]) for row in history]
    actor_loss = [float(row["behavior/actor_loss"]) for row in history]
    critic_loss = [float(row["behavior/critic_loss"]) for row in history]
    imagined_reward = [float(row["behavior/imagined_reward_mean"]) for row in history]
    replay_size = [int(row["replay_size"]) for row in history]

    has_kl = "world_model/kl_loss" in history[0]
    if has_kl:
        world_kl = [float(row["world_model/kl_loss"]) for row in history]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    figure.suptitle(title)

    axes[0, 0].plot(steps, world_total, label="total")
    axes[0, 0].plot(steps, world_recon, label="reconstruction")
    axes[0, 0].plot(steps, world_reward, label="reward")
    if has_kl:
        axes[0, 0].plot(steps, world_kl, label="kl", linestyle="--")
    axes[0, 0].set_title("World-model losses")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, actor_loss, label="actor")
    axes[0, 1].plot(steps, critic_loss, label="critic")
    axes[0, 1].set_title("Behavior losses")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, imagined_reward, label="imagined reward mean", color="tab:green")
    axes[1, 0].set_title("Imagined reward trend")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend()

    axes[1, 1].plot(steps, replay_size, label="replay size", color="tab:purple")
    axes[1, 1].set_title("Replay growth")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].legend()

    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def export_training_history_artifacts(
    metrics_csv: str | Path,
    output_dir: str | Path,
    *,
    prefix: str = "training_history",
) -> dict[str, Path]:
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    history = load_cycle_metrics_history(metrics_csv)
    summary = summarize_training_history(history)
    summary_path = output_directory / f"{prefix}_summary.json"
    plot_path = output_directory / f"{prefix}_curves.png"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_training_history(history, plot_path)
    return {
        "summary": summary_path,
        "curves": plot_path,
    }