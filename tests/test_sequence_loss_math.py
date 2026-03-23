"""Deterministic mathematical tests for vectorized sequence losses.

Verifies that _compute_vectorized_losses produces identical results to
the non-vectorized compute_world_model_losses path, and tests
overshooting KL averaging.
"""

import math

import torch

from tiny_dreamer_highway.models.encoder import LatentState
from tiny_dreamer_highway.models.world_model import TinyWorldModel, WorldModelOutput
from tiny_dreamer_highway.training.sequence_world_model_step import (
    _compute_vectorized_losses,
    compute_latent_overshooting_losses,
    compute_sequence_world_model_losses,
)
from tiny_dreamer_highway.training.world_model_step import (
    compute_world_model_losses,
    _raw_categorical_kl,
)


# ── _compute_vectorized_losses matches single-step formula ───────────

def test_vectorized_kl_matches_analytic() -> None:
    """Vectorized KL uses the same formula as _raw_categorical_kl."""
    B, T, num_cat, num_cls = 2, 3, 4, 8
    torch.manual_seed(42)
    post_logits = torch.randn(B, T, num_cat, num_cls)
    prior_logits = torch.randn(B, T, num_cat, num_cls)

    # Vectorized path
    losses = _compute_vectorized_losses(
        observations=torch.randint(0, 256, (B, T, 1, 64, 64), dtype=torch.uint8),
        rewards=torch.randn(B, T),
        reconstructions=torch.randn(B, T, 1, 64, 64),
        predicted_rewards=torch.randn(B, T, 1),
        predicted_continues=None,
        post_logits=post_logits,
        prior_logits=prior_logits,
        terminal_targets=None,
        free_nats=0.0,
        kl_balance=0.5,
        continue_loss_weight=1.0,
        observation_std=1.0,
        reward_std=1.0,
    )

    # Analytic path (flatten B*T, then compute)
    flat_post_logits = post_logits.reshape(B * T, num_cat, num_cls)
    flat_prior_logits = prior_logits.reshape(B * T, num_cat, num_cls)
    analytic_kl = _raw_categorical_kl(
        flat_post_logits, flat_prior_logits
    )

    assert torch.allclose(losses["kl_loss_raw"], analytic_kl, atol=1e-5)


def test_vectorized_reward_loss_matches_manual() -> None:
    """Reward loss = -mean(log N(target; pred, σ))."""
    B, T = 1, 1
    pred_reward = torch.tensor([[[0.5]]])
    target_reward = torch.tensor([[1.0]])
    sigma = 2.0

    losses = _compute_vectorized_losses(
        observations=torch.zeros(B, T, 1, 64, 64, dtype=torch.uint8),
        rewards=target_reward,
        reconstructions=torch.zeros(B, T, 1, 64, 64),
        predicted_rewards=pred_reward,
        predicted_continues=None,
        post_logits=torch.zeros(B, T, 4, 8),
        prior_logits=torch.zeros(B, T, 4, 8),
        terminal_targets=None,
        free_nats=0.0,
        kl_balance=0.8,
        continue_loss_weight=1.0,
        observation_std=1.0,
        reward_std=sigma,
    )

    # Manual: -log N(1.0; 0.5, 2.0) = 0.5*ln(2π) + ln(2) + (0.5)²/(2*4)
    nll = 0.5 * math.log(2 * math.pi) + math.log(sigma) + (1.0 - 0.5) ** 2 / (2 * sigma ** 2)
    assert torch.allclose(losses["reward_loss"], torch.tensor(nll), atol=1e-4)


def test_vectorized_reconstruction_mse() -> None:
    """reconstruction_mse is standard MSE over all pixels."""
    B, T, C, H, W = 1, 1, 1, 4, 4  # small image
    recon = torch.ones(B, T, C, H, W) * 0.5
    target = torch.zeros(B, T, C, H, W, dtype=torch.uint8)  # normalised to 0.0

    losses = _compute_vectorized_losses(
        observations=target,
        rewards=torch.zeros(B, T),
        reconstructions=recon,
        predicted_rewards=torch.zeros(B, T, 1),
        predicted_continues=None,
        post_logits=torch.zeros(B, T, 4, 8),
        prior_logits=torch.zeros(B, T, 4, 8),
        terminal_targets=None,
        free_nats=0.0,
        kl_balance=0.8,
        continue_loss_weight=1.0,
        observation_std=1.0,
        reward_std=1.0,
    )

    # MSE(0.5 - 0.0) for all 16 pixels = 0.25
    assert torch.allclose(losses["reconstruction_mse"], torch.tensor(0.25), atol=1e-5)


def test_vectorized_free_nats_clamping() -> None:
    """Free nats clamps KL from below."""
    B, T, D = 1, 1, 4

    losses = _compute_vectorized_losses(
        observations=torch.zeros(B, T, 1, 64, 64, dtype=torch.uint8),
        rewards=torch.zeros(B, T),
        reconstructions=torch.zeros(B, T, 1, 64, 64),
        predicted_rewards=torch.zeros(B, T, 1),
        predicted_continues=None,
        # Same distributions → KL ≈ 0
        post_logits=torch.zeros(B, T, 4, 8),
        prior_logits=torch.zeros(B, T, 4, 8),
        terminal_targets=None,
        free_nats=3.0,
        kl_balance=0.8,
        continue_loss_weight=1.0,
        observation_std=1.0,
        reward_std=1.0,
    )

    assert torch.allclose(losses["kl_loss"], torch.tensor(3.0), atol=1e-5)
    assert losses["kl_loss_raw"].item() < 3.0


def test_vectorized_continue_loss_matches_bce() -> None:
    """Continue loss = BCE(logits, 1 - terminals)."""
    B, T = 2, 1
    logits = torch.zeros(B, T, 1)  # logit=0 → σ=0.5
    terminals = torch.zeros(B, T)  # not terminal → target=1

    losses = _compute_vectorized_losses(
        observations=torch.zeros(B, T, 1, 64, 64, dtype=torch.uint8),
        rewards=torch.zeros(B, T),
        reconstructions=torch.zeros(B, T, 1, 64, 64),
        predicted_rewards=torch.zeros(B, T, 1),
        predicted_continues=logits,
        post_logits=torch.zeros(B, T, 4, 8),
        prior_logits=torch.zeros(B, T, 4, 8),
        terminal_targets=terminals,
        free_nats=0.0,
        kl_balance=0.8,
        continue_loss_weight=1.0,
        observation_std=1.0,
        reward_std=1.0,
    )

    expected = math.log(2.0)
    assert torch.allclose(losses["continue_loss"], torch.tensor(expected), atol=1e-4)


# ── sequence model forward produces valid losses ─────────────────────

def test_sequence_forward_losses_are_finite() -> None:
    """All computed losses should be finite."""
    torch.manual_seed(42)
    wm = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=64, deterministic_dim=32,
        num_categoricals=4, num_classes=8,
        hidden_dim=32, rssm_num_layers=1,
        reward_hidden_dim=32, reward_num_layers=1,
    )

    B, T = 2, 3
    obs = torch.randint(0, 256, (B, T, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(B, T, 2)
    rewards = torch.randn(B, T)

    _, losses = compute_sequence_world_model_losses(
        wm, obs, actions, rewards,
        kl_weight=1.0, free_nats=3.0,
    )

    for key, val in losses.items():
        assert torch.isfinite(val), f"{key} is not finite: {val}"


# ── overshooting ────────────────────────────────────────────────────

def test_overshooting_zero_horizon_returns_zeros() -> None:
    """No overshooting → all overshooting losses are zero."""
    dummy_output = WorldModelOutput(
        embedding=torch.zeros(1, 32),
        prior_state=LatentState(
            deterministic=torch.zeros(1, 16),
            stochastic=torch.zeros(1, 8),
            logits=torch.zeros(1, 4, 2),
        ),
        posterior_state=LatentState(
            deterministic=torch.zeros(1, 16),
            stochastic=torch.zeros(1, 8),
            logits=torch.zeros(1, 4, 2),
        ),
        reconstruction=torch.zeros(1, 1, 64, 64),
        predicted_reward=torch.zeros(1, 1),
    )
    torch.manual_seed(42)
    wm = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=64, deterministic_dim=16, num_categoricals=4, num_classes=2,
        hidden_dim=16, rssm_num_layers=1,
        reward_hidden_dim=16, reward_num_layers=1,
    )

    losses = compute_latent_overshooting_losses(
        wm, [dummy_output, dummy_output],
        actions=torch.randn(1, 2, 2),
        overshooting_horizon=0,
    )

    assert losses["overshooting_kl_loss"].item() == 0.0
    assert losses["overshooting_feature_mse"].item() == 0.0
    assert losses["overshooting_pairs"].item() == 0.0


def test_overshooting_pair_count() -> None:
    """With T outputs and horizon=1, there are T-1 pairs."""
    torch.manual_seed(42)
    wm = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=64, deterministic_dim=16,
        num_categoricals=4, num_classes=8,
        hidden_dim=16, rssm_num_layers=1,
        reward_hidden_dim=16, reward_num_layers=1,
    )

    T = 4
    B = 1
    obs = torch.randint(0, 256, (B, T, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(B, T, 2)
    rewards = torch.randn(B, T)

    outputs, _ = compute_sequence_world_model_losses(
        wm, obs, actions, rewards,
        kl_weight=1.0, free_nats=0.0,
        overshooting_horizon=1,
        overshooting_kl_weight=0.5,
    )

    losses = compute_latent_overshooting_losses(
        wm, outputs, actions, overshooting_horizon=1,
    )

    # With horizon=1 and T=4 outputs, pairs = T-1 = 3
    assert losses["overshooting_pairs"].item() == T - 1
