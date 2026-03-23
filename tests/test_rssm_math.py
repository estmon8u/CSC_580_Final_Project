"""Deterministic mathematical tests for RecurrentStateSpaceModel.

Verifies categorical sampling, straight-through gradients, GRU
deterministic state transitions, initial state zeros, and
prior vs posterior divergence.
"""

import torch

from tiny_dreamer_highway.models.encoder import LatentState
from tiny_dreamer_highway.models.rssm import RecurrentStateSpaceModel


def _make_rssm(seed: int = 42, **kwargs) -> RecurrentStateSpaceModel:
    defaults = dict(
        action_dim=2, embedding_dim=64, deterministic_dim=32,
        num_categoricals=4, num_classes=8, hidden_dim=32, num_layers=1,
    )
    defaults.update(kwargs)
    torch.manual_seed(seed)
    return RecurrentStateSpaceModel(**defaults)


# ── categorical sampling ─────────────────────────────────────────────

def test_sample_categorical_returns_correct_shape() -> None:
    """Straight-through sample produces flattened one-hot of expected size."""
    rssm = _make_rssm()
    logits = torch.randn(3, 4, 8)  # (B, num_cat, num_cls)
    result = rssm._sample_categorical(logits)
    assert result.shape == (3, 32)  # 4 * 8 = 32


def test_sample_categorical_is_approximately_one_hot() -> None:
    """Each categorical block should be approximately one-hot (hard sample forward)."""
    rssm = _make_rssm()
    logits = torch.randn(2, 4, 8)
    result = rssm._sample_categorical(logits)
    reshaped = result.reshape(2, 4, 8)
    # Each block should sum to ~1.0 (straight-through adds soft correction)
    sums = reshaped.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_sample_categorical_has_gradients() -> None:
    """Straight-through estimator should allow gradient flow."""
    rssm = _make_rssm()
    logits = torch.randn(2, 4, 8, requires_grad=True)
    result = rssm._sample_categorical(logits)
    result.sum().backward()
    assert logits.grad is not None
    assert logits.grad.shape == (2, 4, 8)


# ── initial_state ────────────────────────────────────────────────────

def test_initial_state_is_all_zeros() -> None:
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=4)

    assert torch.equal(state.deterministic, torch.zeros(4, 32))
    assert torch.equal(state.stochastic, torch.zeros(4, 32))  # 4 * 8 = 32


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
        stochastic=torch.zeros(1, 32),  # 4 * 8 = 32
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
    assert torch.equal(out1.logits, out2.logits)


def test_imagine_step_output_shapes() -> None:
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=3)
    action = torch.randn(3, 2)

    out = rssm.imagine_step(state, action)

    assert out.deterministic.shape == (3, 32)
    assert out.stochastic.shape == (3, 32)  # 4 * 8 = 32
    assert out.logits.shape == (3, 4, 8)


def test_imagine_step_logits_are_finite() -> None:
    """Prior logits should always be finite."""
    rssm = _make_rssm()
    state = rssm.initial_state(batch_size=5)
    action = torch.randn(5, 2)

    out = rssm.imagine_step(state, action)
    assert torch.all(torch.isfinite(out.logits))


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
    # But logits differ because posterior uses embedding
    assert not torch.allclose(prior.logits, posterior.logits)


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
    rssm = _make_rssm(deterministic_dim=32, num_categoricals=4, num_classes=8)
    state = rssm.initial_state(batch_size=1)

    # stochastic_dim = 4 * 8 = 32, deterministic_dim = 32 → total = 64
    assert state.features.shape[-1] == 32 + 32
