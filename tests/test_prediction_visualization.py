from pathlib import Path

import torch

from tiny_dreamer_highway.evaluation import (
    export_prediction_artifacts,
    plot_prediction_metrics,
    save_prediction_comparison_grid,
)


def make_step_metrics() -> list[dict[str, float]]:
    return [
        {"step": 1.0, "mse": 0.15, "psnr": 8.2, "ssim": 0.12},
        {"step": 2.0, "mse": 0.18, "psnr": 7.9, "ssim": 0.09},
        {"step": 3.0, "mse": 0.22, "psnr": 7.1, "ssim": 0.05},
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