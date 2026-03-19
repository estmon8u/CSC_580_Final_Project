"""Deterministic mathematical tests for Critic value network.

Verifies exact numerics: distribution log_prob matches hand-computed
Gaussian NLL, fixed distribution_std, seeded forward-pass determinism,
and value output shape.
"""

import math

import torch

from tiny_dreamer_highway.models.critic import Critic


def _make_critic(seed: int = 42, **kwargs) -> Critic:
    defaults = dict(latent_dim=40, hidden_dim=32, num_layers=2, distribution_std=1.0)
    defaults.update(kwargs)
    torch.manual_seed(seed)
    return Critic(**defaults)


# ── distribution log_prob matches Gaussian formula ───────────────────

def test_distribution_log_prob_zero_residual() -> None:
    """When target == forward output (μ), log_prob = -0.5*ln(2π) - ln(σ).

    With σ=1: log_prob = -0.5*ln(2π) ≈ -0.9189 per action dim.
    Critic has 1 output dim, so Independent(..., 1) sums over that dim.
    """
    critic = _make_critic()

    # Zero all weights so output is 0 for zero input
    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()

    x = torch.zeros(1, 40)
    target = torch.zeros(1, 1)

    dist = critic.distribution(x)
    lp = dist.log_prob(target)

    # log N(0; 0, 1) = -0.5 * ln(2π)
    expected = -0.5 * math.log(2 * math.pi)
    assert torch.allclose(lp, torch.tensor([expected]), atol=1e-4)


def test_distribution_log_prob_nonzero_residual() -> None:
    """log N(v; μ, σ) = -0.5*ln(2π) - ln(σ) - (v-μ)²/(2σ²).

    With μ=0, σ=1, v=2:
      log_prob = -0.5*ln(2π) - 0 - 4/2 = -0.5*ln(2π) - 2.0 ≈ -2.9189
    """
    critic = _make_critic(distribution_std=1.0)

    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()

    x = torch.zeros(1, 40)
    target = torch.tensor([[2.0]])

    dist = critic.distribution(x)
    lp = dist.log_prob(target)

    expected = -0.5 * math.log(2 * math.pi) - 2.0
    assert torch.allclose(lp, torch.tensor([expected]), atol=1e-4)


def test_distribution_log_prob_custom_std() -> None:
    """With σ=2, v=0, μ=0: log_prob = -0.5*ln(2π) - ln(2) ≈ -1.6121."""
    critic = _make_critic(distribution_std=2.0)

    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()

    x = torch.zeros(1, 40)
    target = torch.zeros(1, 1)

    dist = critic.distribution(x)
    lp = dist.log_prob(target)

    expected = -0.5 * math.log(2 * math.pi) - math.log(2.0)
    assert torch.allclose(lp, torch.tensor([expected]), atol=1e-4)


# ── distribution std is fixed ────────────────────────────────────────

def test_distribution_std_is_constant() -> None:
    """The std in the distribution is distribution_std, not learned."""
    critic = _make_critic(distribution_std=3.0)

    x = torch.randn(5, 40)
    dist = critic.distribution(x)
    base_dist = dist.base_dist

    assert torch.allclose(base_dist.scale, torch.full_like(base_dist.scale, 3.0))


# ── eval determinism ─────────────────────────────────────────────────

def test_eval_is_deterministic() -> None:
    critic = _make_critic()
    critic.eval()
    x = torch.randn(3, 40)

    v1 = critic(x)
    v2 = critic(x)
    assert torch.equal(v1, v2)


# ── seeded reproducibility ──────────────────────────────────────────

def test_seeded_critic_is_reproducible() -> None:
    x = torch.randn(3, 40)

    torch.manual_seed(42)
    c1 = Critic(latent_dim=40, hidden_dim=32, num_layers=2)
    out1 = c1(x)

    torch.manual_seed(42)
    c2 = Critic(latent_dim=40, hidden_dim=32, num_layers=2)
    out2 = c2(x)

    assert torch.equal(out1, out2)


# ── output shape ─────────────────────────────────────────────────────

def test_critic_output_scalar_per_batch() -> None:
    critic = _make_critic()
    x = torch.randn(7, 40)
    v = critic(x)
    assert v.shape == (7, 1)


def test_critic_handles_time_dimension() -> None:
    critic = _make_critic()
    x = torch.randn(2, 5, 40)
    v = critic(x)
    assert v.shape == (2, 5, 1)


# ── finite outputs ───────────────────────────────────────────────────

def test_critic_output_is_finite() -> None:
    critic = _make_critic()
    x = torch.randn(10, 40)
    v = critic(x)
    assert torch.all(torch.isfinite(v))
