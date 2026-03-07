from pathlib import Path

import torch

from tiny_dreamer_highway.evaluation import (
    build_prediction_video_frames,
    export_prediction_artifacts,
    export_prediction_media_bundle,
    export_prediction_video,
    plot_prediction_metrics,
    save_prediction_comparison_grid,
)


def make_step_metrics() -> list[dict[str, float]]:
    return [
        {"step": 1.0, "mse": 0.15, "psnr": 8.2, "ssim": 0.12, "nll": 120.0},
        {"step": 2.0, "mse": 0.18, "psnr": 7.9, "ssim": 0.09, "nll": 140.0},
        {"step": 3.0, "mse": 0.22, "psnr": 7.1, "ssim": 0.05, "nll": 150.0},
    ]


def make_prediction_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    predicted = torch.rand(2, 3, 1, 16, 16)
    target = torch.rand(2, 3, 1, 16, 16)
    return predicted, target


def test_plot_prediction_metrics_writes_png(tmp_path: Path) -> None:
    output = plot_prediction_metrics(make_step_metrics(), tmp_path / "metrics.png")

    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_save_prediction_comparison_grid_writes_png(tmp_path: Path) -> None:
    predicted, target = make_prediction_tensors()

    output = save_prediction_comparison_grid(
        predicted,
        target,
        tmp_path / "comparison.png",
        example_index=1,
        max_steps=2,
    )

    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_export_prediction_artifacts_writes_plot_and_grid(tmp_path: Path) -> None:
    predicted, target = make_prediction_tensors()

    outputs = export_prediction_artifacts(
        make_step_metrics(),
        predicted,
        target,
        tmp_path,
        prefix="smoke",
    )

    assert set(outputs.keys()) == {"metrics_plot", "comparison_grid"}
    assert outputs["metrics_plot"].name == "smoke_metrics.png"
    assert outputs["comparison_grid"].name == "smoke_comparison.png"
    assert all(path.exists() for path in outputs.values())


def test_build_prediction_video_frames_returns_rgb_frames() -> None:
    predicted, target = make_prediction_tensors()

    frames = build_prediction_video_frames(predicted, target, example_index=0, max_steps=2)

    assert len(frames) == 2
    assert frames[0].ndim == 3
    assert frames[0].shape[-1] == 3


def test_export_prediction_video_writes_gif(tmp_path: Path) -> None:
    predicted, target = make_prediction_tensors()

    output = export_prediction_video(
        predicted,
        target,
        tmp_path / "comparison.gif",
        max_steps=3,
        fps=3,
    )

    assert output.exists()
    assert output.suffix == ".gif"
    assert output.stat().st_size > 0


def test_export_prediction_media_bundle_writes_plot_grid_and_video(tmp_path: Path) -> None:
    predicted, target = make_prediction_tensors()

    outputs = export_prediction_media_bundle(
        make_step_metrics(),
        predicted,
        target,
        tmp_path,
        prefix="bundle",
        max_steps=2,
        fps=2,
    )

    assert set(outputs.keys()) == {"metrics_plot", "comparison_grid", "comparison_video"}
    assert outputs["comparison_video"].name == "bundle_comparison.gif"
    assert all(path.exists() for path in outputs.values())