import torch

from tiny_dreamer_highway.evaluation import (
    compute_frame_metrics,
    evaluate_latent_rollout_consistency,
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


def test_compute_frame_metrics_includes_nll_when_std_provided() -> None:
    predicted = torch.zeros(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)

    metrics = compute_frame_metrics(predicted, target, observation_std=1.0)

    assert set(metrics.keys()) == {"mse", "psnr", "ssim", "nll"}
    assert metrics["nll"] >= 0.0


def test_rollout_imagined_observations_returns_expected_shape() -> None:
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    seed_observation = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
    future_actions = torch.randn(3, 4, 2)

    predictions = rollout_imagined_observations(model, seed_observation, future_actions)

    assert predictions.shape == (3, 4, 1, 64, 64)


def test_evaluate_n_step_predictions_returns_per_step_metrics_and_summary() -> None:
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
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
    assert set(results["summary"].keys()) == {"mse_mean", "psnr_mean", "ssim_mean", "nll_mean"}
    assert all("nll" in item for item in results["step_metrics"])


def test_evaluate_latent_rollout_consistency_returns_drift_summary() -> None:
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    seed_observation = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    future_actions = torch.randn(2, 3, 2)
    target_observations = torch.randint(0, 256, (2, 3, 1, 64, 64), dtype=torch.uint8)

    results = evaluate_latent_rollout_consistency(
        model,
        seed_observation,
        future_actions,
        target_observations,
    )

    assert len(results["step_metrics"]) == 3
    assert set(results["summary"].keys()) == {
        "deterministic_mse_mean",
        "stochastic_mse_mean",
        "feature_mse_mean",
        "prior_posterior_kl_mean",
    }