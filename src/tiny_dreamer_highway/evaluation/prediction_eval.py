"""N-step prediction evaluation helpers.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import math

import torch
from torch.distributions import Independent, Normal
from torch import Tensor

from tiny_dreamer_highway.models import TinyWorldModel
from tiny_dreamer_highway.training.world_model_step import gaussian_kl_divergence


def _normalize_observations(observations: Tensor, dtype: torch.dtype) -> Tensor:
    normalized = observations.to(dtype=dtype)
    if observations.dtype == torch.uint8:
        normalized = normalized / 255.0
    return normalized.clamp(0.0, 1.0)


def compute_frame_metrics(
    predicted: Tensor,
    target: Tensor,
    *,
    observation_std: float | None = None,
) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have matching shapes")

    predicted = _normalize_observations(predicted, dtype=torch.float32)
    target = _normalize_observations(target, dtype=torch.float32)

    mse = torch.mean((predicted - target) ** 2).item()
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)

    pred_flat = predicted.reshape(predicted.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    mu_x = pred_flat.mean(dim=-1)
    mu_y = target_flat.mean(dim=-1)
    var_x = pred_flat.var(dim=-1, unbiased=False)
    var_y = target_flat.var(dim=-1, unbiased=False)
    cov_xy = ((pred_flat - mu_x.unsqueeze(-1)) * (target_flat - mu_y.unsqueeze(-1))).mean(dim=-1)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2))
        / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2))
    ).mean().item()

    metrics = {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }
    if observation_std is not None:
        if observation_std <= 0:
            raise ValueError("observation_std must be positive")
        raw_predicted = predicted.to(dtype=torch.float32)
        observation_dist = Independent(
            Normal(raw_predicted, torch.full_like(raw_predicted, observation_std)),
            raw_predicted.ndim - 1,
        )
        metrics["nll"] = float(-observation_dist.log_prob(target).mean().item())
    return metrics


def rollout_imagined_observations(
    model: TinyWorldModel,
    seed_observation: Tensor,
    future_actions: Tensor,
) -> Tensor:
    if seed_observation.ndim != 4:
        raise ValueError("seed_observation must have shape (B, C, H, W)")
    if future_actions.ndim != 3:
        raise ValueError("future_actions must have shape (B, H, action_dim)")
    if seed_observation.shape[0] != future_actions.shape[0]:
        raise ValueError("seed_observation and future_actions must share batch size")

    batch_size = seed_observation.shape[0]
    device = seed_observation.device
    zero_action = torch.zeros(batch_size, future_actions.shape[-1], device=device, dtype=future_actions.dtype)

    with torch.no_grad():
        initial_state = model.rssm.initial_state(batch_size=batch_size, device=device)
        embedding = model.encoder.encode(seed_observation)
        state = model.rssm.observe_step(initial_state, zero_action, embedding)

        predictions: list[Tensor] = []
        for step in range(future_actions.shape[1]):
            state = model.rssm.imagine_step(state, future_actions[:, step])
            predictions.append(model.decoder(state.features))

    return torch.stack(predictions, dim=1)


def evaluate_n_step_predictions(
    model: TinyWorldModel,
    seed_observation: Tensor,
    future_actions: Tensor,
    target_observations: Tensor,
) -> dict[str, object]:
    if target_observations.ndim != 5:
        raise ValueError("target_observations must have shape (B, H, C, H, W)")
    if future_actions.shape[:2] != target_observations.shape[:2]:
        raise ValueError("future_actions and target_observations must agree on batch and horizon")

    predicted_observations = rollout_imagined_observations(model, seed_observation, future_actions)
    observation_std = getattr(model.decoder, "distribution_std", None)

    step_metrics: list[dict[str, float]] = []
    for step in range(predicted_observations.shape[1]):
        metrics = compute_frame_metrics(
            predicted_observations[:, step],
            target_observations[:, step],
            observation_std=observation_std,
        )
        metrics["step"] = float(step + 1)
        step_metrics.append(metrics)

    summary = {
        "mse_mean": float(sum(item["mse"] for item in step_metrics) / len(step_metrics)),
        "psnr_mean": float(sum(item["psnr"] for item in step_metrics) / len(step_metrics)),
        "ssim_mean": float(sum(item["ssim"] for item in step_metrics) / len(step_metrics)),
    }
    if all("nll" in item for item in step_metrics):
        summary["nll_mean"] = float(sum(item["nll"] for item in step_metrics) / len(step_metrics))

    return {
        "predictions": predicted_observations,
        "step_metrics": step_metrics,
        "summary": summary,
    }


def evaluate_latent_rollout_consistency(
    model: TinyWorldModel,
    seed_observation: Tensor,
    future_actions: Tensor,
    target_observations: Tensor,
) -> dict[str, object]:
    if seed_observation.ndim != 4:
        raise ValueError("seed_observation must have shape (B, C, H, W)")
    if future_actions.ndim != 3:
        raise ValueError("future_actions must have shape (B, H, action_dim)")
    if target_observations.ndim != 5:
        raise ValueError("target_observations must have shape (B, H, C, H, W)")
    if future_actions.shape[:2] != target_observations.shape[:2]:
        raise ValueError("future_actions and target_observations must agree on batch and horizon")

    batch_size = seed_observation.shape[0]
    device = seed_observation.device
    zero_action = torch.zeros(batch_size, future_actions.shape[-1], device=device, dtype=future_actions.dtype)

    with torch.no_grad():
        initial_state = model.rssm.initial_state(batch_size=batch_size, device=device)
        embedding = model.encoder.encode(seed_observation)
        grounded_state = model.rssm.observe_step(initial_state, zero_action, embedding)
        imagined_state = grounded_state

        step_metrics: list[dict[str, float]] = []
        for step in range(future_actions.shape[1]):
            action = future_actions[:, step]
            imagined_state = model.rssm.imagine_step(imagined_state, action)
            target_embedding = model.encoder.encode(target_observations[:, step])
            grounded_state = model.rssm.observe_step(grounded_state, action, target_embedding)

            deterministic_mse = torch.mean(
                (imagined_state.deterministic - grounded_state.deterministic) ** 2
            ).item()
            stochastic_mse = torch.mean(
                (imagined_state.stochastic - grounded_state.stochastic) ** 2
            ).item()
            feature_mse = torch.mean((imagined_state.features - grounded_state.features) ** 2).item()
            prior_posterior_kl = gaussian_kl_divergence(
                grounded_state.dist_mean,
                grounded_state.dist_std,
                imagined_state.dist_mean,
                imagined_state.dist_std,
            ).item()
            step_metrics.append(
                {
                    "step": float(step + 1),
                    "deterministic_mse": float(deterministic_mse),
                    "stochastic_mse": float(stochastic_mse),
                    "feature_mse": float(feature_mse),
                    "prior_posterior_kl": float(prior_posterior_kl),
                }
            )

    summary = {
        "deterministic_mse_mean": float(sum(item["deterministic_mse"] for item in step_metrics) / len(step_metrics)),
        "stochastic_mse_mean": float(sum(item["stochastic_mse"] for item in step_metrics) / len(step_metrics)),
        "feature_mse_mean": float(sum(item["feature_mse"] for item in step_metrics) / len(step_metrics)),
        "prior_posterior_kl_mean": float(sum(item["prior_posterior_kl"] for item in step_metrics) / len(step_metrics)),
    }
    return {
        "step_metrics": step_metrics,
        "summary": summary,
    }