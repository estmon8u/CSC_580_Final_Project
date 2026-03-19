"""Deterministic mathematical tests for RecurrentStateSpaceModel.

Verifies exact numerics: softplus+min_std distribution parameters,
reparameterised sampling, GRU deterministic state transitions,
initial state zeros, and prior vs posterior divergence.
"""

import math

import torch
import torch.nn.functional as F

from tiny_dreamer_highway.models.encoder import LatentState
from tiny_dreamer_highway.models.rssm import RecurrentStateSpaceModel


def _make_rssm(seed: int = 42, **kwargs) -> RecurrentStateSpaceModel:
    defaults = dict(
        action_dim=2, embedding_dim=64, deterministic_dim=32,
        stochastic_dim=8, hidden_dim=32, min_std=0.1, num_layers=1,
    )
    defaults.update(kwargs)
    torch.manual_seed(seed)
    return RecurrentStateSpaceModel(**defaults)


# ── distribution_parameters ──────────────────────────────────────────

def test_distribution_parameters_softplus_plus_min_std() -> None:
    """std = softplus(raw_std) + min_std.

    softplus(0) = ln(2) ≈ 0.6931
    so std(raw=0) = ln(2) + 0.1 ≈ 0.7931
    """
    rssm = _make_rssm(min_std=0.1)
    # 16 dims = 8 mean + 8 std (stochastic_dim=8)
    stats = torch.zeros(1, 16)  # all zeros → raw_std = 0 for std half
    mean, std = rssm._distribution_parameters(stats)

    assert mean.shape == (1, 8)
    assert std.shape == (1, 8)
    assert torch.allclose(mean, torch.zeros(1, 8))

    expected_std = math.log(2.0) + 0.1  # ≈ 0.7931
    assert torch.allclose(std, torch.full((1, 8), expected_std), atol=1e-5)


def test_distribution_parameters_large_raw_std_saturates() -> None:
    """softplus(x) ≈ x for large x, so std ≈ x + min_std."""
    rssm = _make_rssm(min_std=0.1)
    stats = torch.zeros(1, 16)
    stats[0, 8:] = 10.0  # large raw_std

    _, std = rssm._distribution_parameters(stats)

    # softplus(10) ≈ 10.0000 (saturated), so std ≈ 10.1
    expected = F.softplus(torch.tensor(10.0)) + 0.1
    assert torch.allclose(std, torch.full((1, 8), expected.item()), atol=1e-4)


def test_distribution_parameters_negative_raw_std() -> None:
    """softplus(-5) is small, so std ≈ softplus(-5) + min_std."""
    rssm = _make_rssm(min_std=0.1)
    stats = torch.zeros(1, 16)
    stats[0, 8:] = -5.0

    _, std = rssm._distribution_parameters(stats)

    expected = F.softplus(torch.tensor(-5.0)) + 0.1
    assert torch.allclose(std, torch.full((1, 8), expected.item()), atol=1e-5)


def test_distribution_parameters_preserves_mean_exactly() -> None:
    """mean is the first half of stats, unchanged."""
    rssm = _make_rssm()
    stats = torch.randn(3, 16)
    mean, _ = rssm._distribution_parameters(stats)

    assert torch.equal(mean, stats[:, :8])


# ── sample_stochastic (reparameterisation trick) ─────────────────────

def test_sample_stochastic_reproduces_with_seed() -> None:
    """z = mean + std * noise, deterministic with fixed seed."""
    rssm = _make_rssm()
    mean = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    std = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    torch.manual_seed(99)
    noise = torch.randn_like(mean)
    expected = mean + std * noise

    torch.manual_seed(99)
    result = rssm._sample_stochastic(mean, std)

    assert torch.allclose(result, expected, atol=1e-6)


def test_sample_stochastic_zero_std_returns_mean() -> None:
    """With std=0, sampling returns mean exactly (no noise contribution)."""
    rssm = _make_rssm()
    mean = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    std = torch.zeros_like(mean)

    result = rssm._sample_stochastic(mean, std)
    assert torch.equal(result, mean)


# ── initial_state ────────────────────────────────────────────────────

def test_initial_state_is_all_zeros() -> None:
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=4)

    assert torch.equal(state.deterministic, torch.zeros(4, 32))
    assert torch.equal(state.stochastic, torch.zeros(4, 8))


def test_initial_state_has_correct_dtype() -> None:
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=2)

    assert state.deterministic.dtype == torch.float32
    assert state.stochastic.dtype == torch.float32


# ── imagine_step (prior) ─────────────────────────────────────────────

def test_imagine_step_is_deterministic_with_seed() -> None:
    """Same seed → identical prior rollout."""
    state = LatentState(
        deterministic=torch.zeros(1, 32),
        stochastic=torch.zeros(1, 8),
    )
    action = torch.zeros(1, 2)

    torch.manual_seed(42)
    rssm1 = _make_rssm(seed=42)
    torch.manual_seed(10)
    out1 = rssm1.imagine_step(state, action)

    torch.manual_seed(42)
    rssm2 = _make_rssm(seed=42)
    torch.manual_seed(10)
    out2 = rssm2.imagine_step(state, action)

    assert torch.equal(out1.deterministic, out2.deterministic)
    assert torch.equal(out1.stochastic, out2.stochastic)
    assert torch.equal(out1.dist_mean, out2.dist_mean)
    assert torch.equal(out1.dist_std, out2.dist_std)


def test_imagine_step_output_shapes() -> None:
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=3)
    action = torch.randn(3, 2)

    out = rssm.imagine_step(state, action)

    assert out.deterministic.shape == (3, 32)
    assert out.stochastic.shape == (3, 8)
    assert out.dist_mean.shape == (3, 8)
    assert out.dist_std.shape == (3, 8)


def test_imagine_step_std_is_at_least_min_std() -> None:
    """Prior std is always >= min_std due to softplus + min_std floor."""
    rssm = _make_rssm(min_std=0.1)
    state = rssm.initial_state(batch_size=5)
    action = torch.randn(5, 2)

    out = rssm.imagine_step(state, action)
    assert torch.all(out.dist_std >= 0.1)


# ── observe_step (posterior) ─────────────────────────────────────────

def test_observe_step_differs_from_imagine_step() -> None:
    """Posterior (with embedding) produces different stochastic state than prior."""
    rssm = _make_rssm(seed=42)
    state = rssm.initial_state(batch_size=2)
    action = torch.randn(2, 2)
    embedding = torch.randn(2, 64)

    torch.manual_seed(0)
    prior = rssm.imagine_step(state, action)

    torch.manual_seed(0)
    posterior = rssm.observe_step(state, action, embedding)

    # Deterministic state is the same (same GRU computation)
    assert torch.allclose(prior.deterministic, posterior.deterministic, atol=1e-6)
    # But stochastic state differs because posterior uses embedding
    assert not torch.allclose(prior.stochastic, posterior.stochastic)


def test_observe_step_stores_embedding() -> None:
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=1)
    action = torch.randn(1, 2)
    embedding = torch.randn(1, 64)

    out = rssm.observe_step(state, action, embedding)
    assert out.embedding is not None
    assert torch.allclose(out.embedding, embedding.to(out.embedding.dtype))


# ── imagine_rollout ──────────────────────────────────────────────────

def test_imagine_rollout_length_matches_action_steps() -> None:
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=2)
    actions = torch.randn(2, 7, 2)  # 7 time steps

    rollout = rssm.imagine_rollout(state, actions)
    assert len(rollout) == 7


def test_imagine_rollout_matches_sequential_steps() -> None:
    """Rollout should equal calling imagine_step in a loop."""
    torch.manual_seed(42)
    rssm = _make_rssm(seed=42)
    state = rssm.initial_state(batch_size=2)
    actions = torch.randn(2, 3, 2)

    # Via rollout
    torch.manual_seed(100)
    rollout = rssm.imagine_rollout(state, actions)

    # Via manual loop
    torch.manual_seed(100)
    manual_state = state
    for t in range(3):
        manual_state = rssm.imagine_step(manual_state, actions[:, t])
        assert torch.allclose(rollout[t].deterministic, manual_state.deterministic, atol=1e-6)
        assert torch.allclose(rollout[t].stochastic, manual_state.stochastic, atol=1e-6)


# ── LatentState.features ─────────────────────────────────────────────

def test_latent_state_features_concatenation_order() -> None:
    """features = cat([stochastic, deterministic], dim=-1)."""
    det = torch.tensor([[1.0, 2.0, 3.0]])
    stoch = torch.tensor([[4.0, 5.0]])

    state = LatentState(deterministic=det, stochastic=stoch)
    expected = torch.tensor([[4.0, 5.0, 1.0, 2.0, 3.0]])

    assert torch.equal(state.features, expected)


def test_latent_state_features_dim_equals_sum() -> None:
    """features dim = stochastic_dim + deterministic_dim."""
    rssm = _make_rssm(deterministic_dim=32, stochastic_dim=8)
    state = rssm.initial_state(batch_size=1)

    assert state.features.shape[-1] == 32 + 8
