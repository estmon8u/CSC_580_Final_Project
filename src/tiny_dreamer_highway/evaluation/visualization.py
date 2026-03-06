"""Visualization helpers for n-step prediction artifacts.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def _normalize_image(image: Tensor) -> Tensor:
    normalized = image.detach().cpu().to(dtype=torch.float32)
    if image.dtype == torch.uint8:
        normalized = normalized / 255.0
    return normalized.clamp(0.0, 1.0)


def _to_display_image(image: Tensor) -> Tensor:
    normalized = _normalize_image(image)
    if normalized.ndim == 2:
        return normalized
    if normalized.ndim != 3:
        raise ValueError("image must have shape (H, W), (C, H, W), or (H, W, C)")

    if normalized.shape[0] in {1, 3, 4}:
        normalized = normalized.movedim(0, -1)
    elif normalized.shape[-1] not in {1, 3, 4}:
        raise ValueError("image must use one, three, or four channels")

    if normalized.shape[-1] == 1:
        normalized = normalized[..., 0]
    return normalized


def _to_uint8_image(image: Tensor) -> np.ndarray:
    display_image = _to_display_image(image)
    image_array = (display_image.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[..., None], 3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = np.repeat(image_array, 3, axis=-1)
    return image_array


def plot_prediction_metrics(
    step_metrics: Sequence[dict[str, float]],
    output_path: str | Path,
    *,
    title: str = "N-step prediction metrics",
) -> Path:
    if not step_metrics:
        raise ValueError("step_metrics must not be empty")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(item["step"]) for item in step_metrics]
    mse = [float(item["mse"]) for item in step_metrics]
    psnr = [float(item["psnr"]) for item in step_metrics]
    ssim = [float(item["ssim"]) for item in step_metrics]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    series = [
        ("MSE", mse, "tab:red"),
        ("PSNR", psnr, "tab:blue"),
        ("SSIM", ssim, "tab:green"),
    ]
    for axis, (label, values, color) in zip(axes, series, strict=True):
        axis.plot(steps, values, marker="o", color=color)
        axis.set_title(label)
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Value")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_prediction_comparison_grid(
    predicted: Tensor,
    target: Tensor,
    output_path: str | Path,
    *,
    example_index: int = 0,
    max_steps: int | None = None,
    title: str = "Predicted vs target frames",
) -> Path:
    if predicted.ndim != 5 or target.ndim != 5:
        raise ValueError("predicted and target must have shape (B, T, C, H, W)")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have matching shapes")
    if not 0 <= example_index < predicted.shape[0]:
        raise IndexError("example_index is out of range")

    horizon = predicted.shape[1] if max_steps is None else min(predicted.shape[1], max_steps)
    if horizon <= 0:
        raise ValueError("at least one prediction step is required")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(horizon, 3, figsize=(9, 3 * horizon), squeeze=False)
    column_titles = ["Target", "Predicted", "Absolute error"]

    for row in range(horizon):
        target_frame = _to_display_image(target[example_index, row])
        predicted_frame = _to_display_image(predicted[example_index, row])
        error_frame = torch.abs(_normalize_image(predicted[example_index, row]) - _normalize_image(target[example_index, row]))
        error_frame = _to_display_image(error_frame)

        frames = [target_frame, predicted_frame, error_frame]
        cmaps = ["gray", "gray", "magma"]
        for column, (axis, frame, cmap) in enumerate(zip(axes[row], frames, cmaps, strict=True)):
            if row == 0:
                axis.set_title(column_titles[column])
            axis.imshow(frame.numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
            axis.axis("off")

        axes[row, 0].set_ylabel(f"Step {row + 1}", rotation=90, labelpad=12)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def export_prediction_artifacts(
    step_metrics: Sequence[dict[str, float]],
    predicted: Tensor,
    target: Tensor,
    output_dir: str | Path,
    *,
    example_index: int = 0,
    prefix: str = "n_step_eval",
    max_steps: int | None = None,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    metrics_plot = plot_prediction_metrics(
        step_metrics,
        directory / f"{prefix}_metrics.png",
    )
    comparison_grid = save_prediction_comparison_grid(
        predicted,
        target,
        directory / f"{prefix}_comparison.png",
        example_index=example_index,
        max_steps=max_steps,
    )
    return {
        "metrics_plot": metrics_plot,
        "comparison_grid": comparison_grid,
    }


def build_prediction_video_frames(
    predicted: Tensor,
    target: Tensor,
    *,
    example_index: int = 0,
    max_steps: int | None = None,
) -> list[np.ndarray]:
    if predicted.ndim != 5 or target.ndim != 5:
        raise ValueError("predicted and target must have shape (B, T, C, H, W)")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have matching shapes")
    if not 0 <= example_index < predicted.shape[0]:
        raise IndexError("example_index is out of range")

    horizon = predicted.shape[1] if max_steps is None else min(predicted.shape[1], max_steps)
    if horizon <= 0:
        raise ValueError("at least one prediction step is required")

    frames: list[np.ndarray] = []
    for step in range(horizon):
        target_frame = _to_uint8_image(target[example_index, step])
        predicted_frame = _to_uint8_image(predicted[example_index, step])
        error_frame = _to_uint8_image(
            torch.abs(_normalize_image(predicted[example_index, step]) - _normalize_image(target[example_index, step]))
        )

        separator = np.full((target_frame.shape[0], 6, 3), 255, dtype=np.uint8)
        stacked_frame = np.concatenate(
            [target_frame, separator, predicted_frame, separator, error_frame],
            axis=1,
        )
        frames.append(stacked_frame)

    return frames


def export_prediction_video(
    predicted: Tensor,
    target: Tensor,
    output_path: str | Path,
    *,
    example_index: int = 0,
    max_steps: int | None = None,
    fps: int = 2,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = build_prediction_video_frames(
        predicted,
        target,
        example_index=example_index,
        max_steps=max_steps,
    )
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    return path


def export_prediction_media_bundle(
    step_metrics: Sequence[dict[str, float]],
    predicted: Tensor,
    target: Tensor,
    output_dir: str | Path,
    *,
    example_index: int = 0,
    prefix: str = "n_step_eval",
    max_steps: int | None = None,
    fps: int = 2,
) -> dict[str, Path]:
    outputs = export_prediction_artifacts(
        step_metrics,
        predicted,
        target,
        output_dir,
        example_index=example_index,
        prefix=prefix,
        max_steps=max_steps,
    )
    video_path = export_prediction_video(
        predicted,
        target,
        Path(output_dir) / f"{prefix}_comparison.gif",
        example_index=example_index,
        max_steps=max_steps,
        fps=fps,
    )
    outputs["comparison_video"] = video_path
    return outputs