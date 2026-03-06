import torch

from tiny_dreamer_highway.evaluation import (
    compute_frame_metrics,
    evaluate_n_step_predictions,
    rollout_imagined_observations,
)
from tiny_dreamer_highway.models import TinyWorldModel


def test_compute_frame_metrics_returns_expected_keys() -> None:
    predicted = torch.zeros(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)

    metrics = compute_frame_metrics(predicted, target)

    assert set(metrics.keys()) == {"mse", "psnr", "ssim"}
    assert metrics["mse"] >= 0.0


def test_rollout_imagined_observations_returns_expected_shape() -> None:
    model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    seed_observation = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
    future_actions = torch.randn(3, 4, 2)

    predictions = rollout_imagined_observations(model, seed_observation, future_actions)

    assert predictions.shape == (3, 4, 1, 64, 64)


def test_evaluate_n_step_predictions_returns_per_step_metrics_and_summary() -> None:
    model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    seed_observation = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    future_actions = torch.randn(2, 3, 2)
    target_observations = torch.randint(0, 256, (2, 3, 1, 64, 64), dtype=torch.uint8)

    results = evaluate_n_step_predictions(
        model,
        seed_observation,
        future_actions,
        target_observations,
    )

    assert results["predictions"].shape == (2, 3, 1, 64, 64)
    assert len(results["step_metrics"]) == 3
    assert set(results["summary"].keys()) == {"mse_mean", "psnr_mean", "ssim_mean"}