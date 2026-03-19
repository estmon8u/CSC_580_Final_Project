"""Metrics export helpers for training artifacts.

Provides functions to flatten per-cycle metrics into flat dicts and
persist them as JSONL, CSV, and summary JSON files.  The CSV writer
handles schema evolution (new metric keys appearing mid-training) by
rewriting the header when necessary.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tiny_dreamer_highway.training.pipeline import PipelineCycleMetrics


def flatten_cycle_metrics(step: int, metrics: PipelineCycleMetrics) -> dict[str, float | int]:
    """Flatten a ``PipelineCycleMetrics`` into a single-level dict.

    Prefixes world-model keys with ``world_model/``, behavior keys with
    ``behavior/``, and evaluation keys with ``evaluation/`` for clean
    CSV column names.
    """
    if step < 0:
        raise ValueError("step must be non-negative")

    record: dict[str, float | int] = {
        "step": step,
        "warm_start_added": metrics.warm_start_added,
        "policy_added": metrics.policy_added,
        "replay_size": metrics.replay_size,
    }
    record.update({f"world_model/{name}": value for name, value in metrics.world_model_metrics.items()})
    record.update({f"behavior/{name}": value for name, value in metrics.behavior_metrics.items()})
    record.update({f"evaluation/{name}": value for name, value in metrics.evaluation_metrics.items()})
    return record


def append_metrics_jsonl(log_file: str | Path, record: dict[str, Any]) -> Path:
    """Append a single metrics record as one JSON line."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def append_metrics_csv(log_file: str | Path, record: dict[str, Any]) -> Path:
    """Append a metrics record to a CSV file, handling schema evolution.

    If new keys appear that weren’t in the original header, the file is
    rewritten with an expanded header before appending the new row.
    """
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(record.keys())
    file_exists = path.exists() and path.stat().st_size > 0

    if file_exists:
        # Read only the header line to check if columns need updating.
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_fields = next(csv.reader(handle), [])
        # Merge: keep existing order, append any new keys at the end.
        merged = list(existing_fields)
        for key in fieldnames:
            if key not in merged:
                merged.append(key)
        fieldnames = merged

        if fieldnames != existing_fields:
            # New columns appeared — rewrite the file with the updated header.
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                existing_rows = list(reader)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({name: existing_row.get(name, "") for name in fieldnames})

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    return path


def write_artifact_summary(
    summary_file: str | Path,
    *,
    step: int,
    record: dict[str, Any],
    checkpoint_file: str | Path | None = None,
) -> Path:
    """Write a JSON file with the latest step, metrics, and checkpoint path."""
    path = Path(summary_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "latest_step": step,
        "latest_metrics": record,
        "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else None,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def export_cycle_metrics(
    log_dir: str | Path,
    *,
    step: int,
    metrics: PipelineCycleMetrics,
    checkpoint_file: str | Path | None = None,
) -> dict[str, Path]:
    """Export all metrics artifacts for one training cycle.

    Writes three files: JSONL (append-only log), CSV (tabular log),
    and a summary JSON with the latest checkpoint pointer.
    """
    directory = Path(log_dir)
    record = flatten_cycle_metrics(step, metrics)
    jsonl_path = append_metrics_jsonl(directory / "cycle_metrics.jsonl", record)
    csv_path = append_metrics_csv(directory / "cycle_metrics.csv", record)
    summary_path = write_artifact_summary(
        directory / "latest_summary.json",
        step=step,
        record=record,
        checkpoint_file=checkpoint_file,
    )
    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "summary": summary_path,
    }