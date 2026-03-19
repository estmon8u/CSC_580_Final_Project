"""Deterministic mathematical tests for all loss functions.

Verifies exact hand-computed values for: Gaussian KL divergence,
reconstruction (Gaussian NLL), reward loss, continue loss (BCE),
free-nats clamping, and total loss composition.
"""

import math

import torch
import torch.nn.functional as F

from tiny_dreamer_highway.models.encoder import LatentState
from tiny_dreamer_highway.models.world_model import WorldModelOutput
from tiny_dreamer_highway.training.world_model_step import (
    compute_world_model_losses,
    gaussian_kl_divergence,
)


# ── Gaussian KL divergence ───────────────────────────────────────────

def test_kl_identical_distributions_is_zero() -> None:
    """KL(N(0,1) || N(0,1)) = 0."""
    mean = torch.zeros(4, 8)
    std = torch.ones(4, 8)
    kl = gaussian_kl_divergence(mean, std, mean, std)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_shifted_mean_unit_variance() -> None:
    """KL(N(1,1) || N(0,1)) = 0.5 * (1+0-1-0) = 0.5.

    var_ratio = 1, mean_diff = 1, per_dim = 0.5*(1+1-1-0) = 0.5
    sum over 1 dim, mean over batch=1 → 0.5
    """
    post_mean = torch.tensor([[1.0]])
    post_std = torch.tensor([[1.0]])
    prior_mean = torch.tensor([[0.0]])
    prior_std = torch.tensor([[1.0]])

    kl = gaussian_kl_divergence(post_mean, post_std, prior_mean, prior_std)
    assert torch.allclose(kl, torch.tensor(0.5), atol=1e-5)


def test_kl_different_variance() -> None:
    """KL(N(0,2) || N(0,1)) = 0.5 * (4 + 0 - 1 - ln(4)).

    per_dim = 0.5 * (4 - 1 - ln4) = 0.5 * (3 - 1.3863) = 0.8069
    """
    post_mean = torch.tensor([[0.0]])
    post_std = torch.tensor([[2.0]])
    prior_mean = torch.tensor([[0.0]])
    prior_std = torch.tensor([[1.0]])

    kl = gaussian_kl_divergence(post_mean, post_std, prior_mean, prior_std)
    expected = 0.5 * (4.0 + 0.0 - 1.0 - math.log(4.0))
    assert torch.allclose(kl, torch.tensor(expected), atol=1e-4)


def test_kl_multiple_dims() -> None:
    """KL sums over latent dims, then averages over batch.

    2 dims, each with KL=0.5 → sum=1.0 per batch item.
    """
    post_mean = torch.tensor([[1.0, 1.0]])
    post_std = torch.tensor([[1.0, 1.0]])
    prior_mean = torch.tensor([[0.0, 0.0]])
    prior_std = torch.tensor([[1.0, 1.0]])

    kl = gaussian_kl_divergence(post_mean, post_std, prior_mean, prior_std)
    # Each dim contributes 0.5, sum over 2 dims = 1.0
    assert torch.allclose(kl, torch.tensor(1.0), atol=1e-5)


def test_kl_batch_averaging() -> None:
    """KL averages over batch. Two items with KL 0.5 each → mean 0.5."""
    post_mean = torch.tensor([[1.0], [1.0]])
    post_std = torch.tensor([[1.0], [1.0]])
    prior_mean = torch.tensor([[0.0], [0.0]])
    prior_std = torch.tensor([[1.0], [1.0]])

    kl = gaussian_kl_divergence(post_mean, post_std, prior_mean, prior_std)
    assert torch.allclose(kl, torch.tensor(0.5), atol=1e-5)


def test_kl_is_always_non_negative() -> None:
    """KL divergence ≥ 0 (information inequality)."""
    torch.manual_seed(42)
    for _ in range(10):
        post_mean = torch.randn(5, 8)
        post_std = torch.rand(5, 8) + 0.01
        prior_mean = torch.randn(5, 8)
        prior_std = torch.rand(5, 8) + 0.01

        kl = gaussian_kl_divergence(post_mean, post_std, prior_mean, prior_std)
        assert kl.item() >= -1e-6  # allow tiny float imprecision


# ── Continue loss (BCE) ──────────────────────────────────────────────

def test_bce_logit_zero_target_one() -> None:
    """BCE(logit=0, target=1) = ln(2) ≈ 0.6931.

    σ(0) = 0.5, so BCE = -[1*ln(0.5) + 0*ln(0.5)] = -ln(0.5) = ln(2)
    """
    logit = torch.tensor([0.0])
    target = torch.tensor([1.0])
    loss = F.binary_cross_entropy_with_logits(logit, target)

    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-5)


def test_bce_large_positive_logit_target_one() -> None:
    """BCE(logit=100, target=1) ≈ 0 (very confident correct prediction)."""
    logit = torch.tensor([100.0])
    target = torch.tensor([1.0])
    loss = F.binary_cross_entropy_with_logits(logit, target)

    assert loss.item() < 1e-4


def test_bce_large_negative_logit_target_zero() -> None:
    """BCE(logit=-100, target=0) ≈ 0 (correctly predicts not-continue)."""
    logit = torch.tensor([-100.0])
    target = torch.tensor([0.0])
    loss = F.binary_cross_entropy_with_logits(logit, target)

    assert loss.item() < 1e-4


# ── Free nats clamping ───────────────────────────────────────────────

def test_free_nats_clamps_low_kl() -> None:
    """raw_kl=2, free_nats=3 → kl_loss=3."""
    raw_kl = torch.tensor(2.0)
    kl_loss = torch.clamp(raw_kl, min=3.0)
    assert kl_loss.item() == 3.0


def test_free_nats_passes_high_kl() -> None:
    """raw_kl=5, free_nats=3 → kl_loss=5."""
    raw_kl = torch.tensor(5.0)
    kl_loss = torch.clamp(raw_kl, min=3.0)
    assert kl_loss.item() == 5.0


# ── Gaussian NLL (reconstruction / reward loss formula) ──────────────

def test_gaussian_nll_zero_residual() -> None:
    """−log N(x; x, σ=1) = 0.5*ln(2π) per dimension.

    For a single scalar: -log_prob = 0.5*ln(2π) ≈ 0.9189.
    """
    mu = torch.tensor([0.5])
    target = torch.tensor([0.5])
    sigma = 1.0

    # Manual: -log N(0.5; 0.5, 1) = 0.5*ln(2π) + 0
    expected_nll = 0.5 * math.log(2 * math.pi)
    actual_nll = 0.5 * math.log(2 * math.pi * sigma**2) + (target - mu).pow(2) / (2 * sigma**2)

    assert math.isclose(actual_nll.item(), expected_nll, rel_tol=1e-6)


def test_gaussian_nll_with_residual() -> None:
    """−log N(0.5; 0.0, 1) = 0.5*ln(2π) + 0.5² / 2 = 0.5*ln(2π) + 0.125."""
    mu = torch.tensor([0.0])
    target = torch.tensor([0.5])
    sigma = 1.0

    expected_nll = 0.5 * math.log(2 * math.pi) + 0.125
    actual_nll = 0.5 * math.log(2 * math.pi * sigma**2) + (target - mu).pow(2) / (2 * sigma**2)

    assert math.isclose(actual_nll.item(), expected_nll, abs_tol=1e-6)


# ── compute_world_model_losses integration ───────────────────────────

def _make_dummy_world_model_output(
    batch_size: int = 2,
    stochastic_dim: int = 4,
    obs_shape: tuple[int, int, int] = (1, 64, 64),
) -> WorldModelOutput:
    """Create a WorldModelOutput with known, simple values."""
    recon = torch.full((batch_size, *obs_shape), 0.5)  # reconstruction μ = 0.5
    pred_reward = torch.zeros(batch_size, 1)
    pred_continue = torch.zeros(batch_size, 1)  # logit = 0 → P(continue)=0.5

    posterior = LatentState(
        deterministic=torch.zeros(batch_size, 8),
        stochastic=torch.zeros(batch_size, stochastic_dim),
        dist_mean=torch.zeros(batch_size, stochastic_dim),
        dist_std=torch.ones(batch_size, stochastic_dim),
    )
    prior = LatentState(
        deterministic=torch.zeros(batch_size, 8),
        stochastic=torch.zeros(batch_size, stochastic_dim),
        dist_mean=torch.zeros(batch_size, stochastic_dim),
        dist_std=torch.ones(batch_size, stochastic_dim),
    )

    return WorldModelOutput(
        embedding=torch.zeros(batch_size, 32),
        prior_state=prior,
        posterior_state=posterior,
        reconstruction=recon,
        predicted_reward=pred_reward,
        predicted_observation_std=1.0,
        predicted_reward_std=1.0,
        predicted_continue=pred_continue,
    )


def test_total_loss_is_sum_of_components() -> None:
    """total = recon + reward + kl_weight*kl + continue_weight*continue."""
    output = _make_dummy_world_model_output()
    target_obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    target_rewards = torch.zeros(2)
    target_terminals = torch.zeros(2)

    losses = compute_world_model_losses(
        output, target_obs, target_rewards,
        kl_weight=2.0, free_nats=0.0,
        target_terminals=target_terminals,
        continue_loss_weight=3.0,
    )

    expected_total = (
        losses["reconstruction_loss"]
        + losses["reward_loss"]
        + 2.0 * losses["kl_loss"]
        + 3.0 * losses["continue_loss"]
    )
    assert torch.allclose(losses["total_loss"], expected_total, atol=1e-5)


def test_kl_loss_is_zero_for_matching_prior_posterior() -> None:
    """When prior == posterior distributions, KL = 0, clamped by free_nats."""
    output = _make_dummy_world_model_output()
    target_obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    target_rewards = torch.zeros(2)

    losses = compute_world_model_losses(
        output, target_obs, target_rewards,
        kl_weight=1.0, free_nats=0.0,
    )

    # KL=0 when prior==posterior, and free_nats=0 doesn't clamp anything
    assert torch.allclose(losses["kl_loss_raw"], torch.tensor(0.0), atol=1e-5)


def test_kl_loss_clamped_by_free_nats() -> None:
    """When raw KL < free_nats, kl_loss = free_nats."""
    output = _make_dummy_world_model_output()
    target_obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    target_rewards = torch.zeros(2)

    losses = compute_world_model_losses(
        output, target_obs, target_rewards,
        kl_weight=1.0, free_nats=3.0,
    )

    # raw KL ≈ 0, clamped to 3.0
    assert torch.allclose(losses["kl_loss"], torch.tensor(3.0), atol=1e-5)
    assert losses["kl_loss_raw"].item() < 3.0


def test_continue_loss_matches_bce_formula() -> None:
    """Continue loss = BCE(logits, 1 - terminals).

    With logit=0, terminal=0 → target=1 → BCE = ln(2) ≈ 0.6931.
    """
    output = _make_dummy_world_model_output()
    target_obs = torch.full((2, 1, 64, 64), 128, dtype=torch.uint8)
    target_rewards = torch.zeros(2)
    target_terminals = torch.zeros(2)  # not terminal → continue target = 1

    losses = compute_world_model_losses(
        output, target_obs, target_rewards,
        kl_weight=1.0, free_nats=0.0,
        target_terminals=target_terminals,
        continue_loss_weight=1.0,
    )

    expected_continue = math.log(2.0)
    assert torch.allclose(losses["continue_loss"], torch.tensor(expected_continue), atol=1e-4)


def test_reward_loss_is_gaussian_nll() -> None:
    """Reward loss = -E[log N(reward; pred_reward, σ)].

    With pred=0, target=0, σ=1: NLL = 0.5*ln(2π) ≈ 0.9189.
    """
    output = _make_dummy_world_model_output()
    target_obs = torch.full((2, 1, 64, 64), 128, dtype=torch.uint8)
    target_rewards = torch.zeros(2)

    losses = compute_world_model_losses(
        output, target_obs, target_rewards,
        kl_weight=1.0, free_nats=0.0,
    )

    expected_reward_nll = 0.5 * math.log(2 * math.pi)
    assert torch.allclose(losses["reward_loss"], torch.tensor(expected_reward_nll), atol=1e-4)


def test_observations_normalised_from_uint8() -> None:
    """uint8 targets are normalised to [0,1] before loss computation."""
    output = _make_dummy_world_model_output()
    # All-black image → normalised to 0.0
    target_obs = torch.zeros(2, 1, 64, 64, dtype=torch.uint8)
    target_rewards = torch.zeros(2)

    losses1 = compute_world_model_losses(
        output, target_obs, target_rewards, kl_weight=1.0, free_nats=0.0,
    )

    # All-white image → normalised to 1.0
    target_obs_white = torch.full((2, 1, 64, 64), 255, dtype=torch.uint8)
    losses2 = compute_world_model_losses(
        output, target_obs_white, target_rewards, kl_weight=1.0, free_nats=0.0,
    )

    # Recon is 0.5 for both; black (0.0) has residual 0.5² = 0.25/pixel,
    # white (1.0) also has residual 0.5² = 0.25/pixel → same MSE.
    assert torch.allclose(
        losses1["reconstruction_mse"], losses2["reconstruction_mse"], atol=1e-5
    )
