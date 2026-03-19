"""Deterministic mathematical tests for TinyWorldModel forward pass.

Verifies seeded forward-pass determinism, prior ≠ posterior divergence,
reconstruction value range, and single-step consistency.
"""

import torch

from tiny_dreamer_highway.models.encoder import LatentState
from tiny_dreamer_highway.models.world_model import TinyWorldModel


def _make_world_model(seed: int = 42, **kwargs) -> TinyWorldModel:
    defaults = dict(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=64, deterministic_dim=32, stochastic_dim=8,
        hidden_dim=32, rssm_num_layers=1,
        reward_hidden_dim=32, reward_num_layers=1,
        continue_hidden_dim=32, continue_num_layers=1,
    )
    defaults.update(kwargs)
    torch.manual_seed(seed)
    return TinyWorldModel(**defaults)


# ── seeded forward-pass determinism ──────────────────────────────────

def test_forward_is_deterministic_with_seed() -> None:
    """Same seed + same inputs → identical outputs."""
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 2)

    torch.manual_seed(42)
    wm1 = _make_world_model(seed=42)
    torch.manual_seed(99)
    out1 = wm1(obs, actions)

    torch.manual_seed(42)
    wm2 = _make_world_model(seed=42)
    torch.manual_seed(99)
    out2 = wm2(obs, actions)

    assert torch.equal(out1.embedding, out2.embedding)
    assert torch.equal(out1.reconstruction, out2.reconstruction)
    assert torch.equal(out1.predicted_reward, out2.predicted_reward)
    assert torch.equal(out1.posterior_state.stochastic, out2.posterior_state.stochastic)
    assert torch.equal(out1.prior_state.stochastic, out2.prior_state.stochastic)


# ── prior ≠ posterior ────────────────────────────────────────────────

def test_prior_and_posterior_deterministic_states_match() -> None:
    """Both prior and posterior use the same GRU step → same deterministic state."""
    wm = _make_world_model()
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 2)

    out = wm(obs, actions)

    assert torch.allclose(
        out.prior_state.deterministic,
        out.posterior_state.deterministic,
        atol=1e-6,
    )


def test_prior_and_posterior_stochastic_states_differ() -> None:
    """Posterior uses obs embedding → different stochastic state from prior."""
    wm = _make_world_model(seed=42)
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 2)

    out = wm(obs, actions)

    # Prior uses only deterministic state, posterior also uses embedding
    # Their distribution parameters should differ
    assert not torch.allclose(
        out.prior_state.dist_mean, out.posterior_state.dist_mean
    )


def test_prior_and_posterior_have_distribution_params() -> None:
    """Both states carry dist_mean and dist_std."""
    wm = _make_world_model()
    obs = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(1, 2)

    out = wm(obs, actions)

    for state in [out.prior_state, out.posterior_state]:
        assert state.dist_mean is not None
        assert state.dist_std is not None
        assert state.dist_mean.shape == (1, 8)  # stochastic_dim=8
        assert state.dist_std.shape == (1, 8)


# ── latent dimensions ────────────────────────────────────────────────

def test_features_dim_equals_det_plus_stoch() -> None:
    """posterior_state.features has dim = deterministic_dim + stochastic_dim."""
    wm = _make_world_model(deterministic_dim=32, stochastic_dim=8)
    obs = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(1, 2)

    out = wm(obs, actions)
    assert out.posterior_state.features.shape[-1] == 32 + 8


# ── reconstruction output ───────────────────────────────────────────

def test_reconstruction_shape_matches_input() -> None:
    """Reconstruction is same spatial shape as input observation."""
    wm = _make_world_model(observation_shape=(1, 64, 64))
    obs = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(3, 2)

    out = wm(obs, actions)
    assert out.reconstruction.shape == (3, 1, 64, 64)


def test_reconstruction_is_finite() -> None:
    wm = _make_world_model()
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 2)

    out = wm(obs, actions)
    assert torch.all(torch.isfinite(out.reconstruction))


# ── reward prediction ───────────────────────────────────────────────

def test_predicted_reward_shape() -> None:
    wm = _make_world_model()
    obs = torch.randint(0, 256, (4, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(4, 2)

    out = wm(obs, actions)
    assert out.predicted_reward.shape == (4, 1)


def test_predicted_reward_is_finite() -> None:
    wm = _make_world_model()
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 2)

    out = wm(obs, actions)
    assert torch.all(torch.isfinite(out.predicted_reward))


# ── continue prediction ─────────────────────────────────────────────

def test_continue_prediction_present_by_default() -> None:
    wm = _make_world_model(use_continue_model=True)
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 2)

    out = wm(obs, actions)
    assert out.predicted_continue is not None
    assert out.predicted_continue.shape == (2, 1)


def test_continue_prediction_absent_when_disabled() -> None:
    wm = _make_world_model(use_continue_model=False)
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 2)

    out = wm(obs, actions)
    assert out.predicted_continue is None


# ── distribution std passthrough ─────────────────────────────────────

def test_observation_std_passed_through() -> None:
    wm = _make_world_model(observation_distribution_std=2.5)
    obs = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(1, 2)

    out = wm(obs, actions)
    assert out.predicted_observation_std == 2.5


def test_reward_std_passed_through() -> None:
    wm = _make_world_model(reward_distribution_std=0.3)
    obs = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(1, 2)

    out = wm(obs, actions)
    assert out.predicted_reward_std == 0.3


# ── embedding ────────────────────────────────────────────────────────

def test_embedding_shape() -> None:
    wm = _make_world_model(embedding_dim=64)
    obs = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(3, 2)

    out = wm(obs, actions)
    assert out.embedding.shape == (3, 64)


# ── prev_state carries forward ──────────────────────────────────────

def test_prev_state_influences_output() -> None:
    """Non-zero prev_state produces different output than zero prev_state."""
    wm = _make_world_model(seed=42)
    obs = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(1, 2)

    torch.manual_seed(10)
    out_default = wm(obs, actions, prev_state=None)

    custom_state = LatentState(
        deterministic=torch.randn(1, 32),
        stochastic=torch.randn(1, 8),
    )
    torch.manual_seed(10)
    out_custom = wm(obs, actions, prev_state=custom_state)

    # Different prev_state → different posterior
    assert not torch.allclose(
        out_default.posterior_state.stochastic,
        out_custom.posterior_state.stochastic,
    )
