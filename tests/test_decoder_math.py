"""Deterministic mathematical tests for ObservationDecoder, RewardPredictor, ContinuePredictor.

Verifies exact numerics: ConvTranspose2d spatial upsampling formula,
distribution std is fixed (not learned), projection dimensions, and
seeded forward-pass reproducibility.
"""

import torch
from torch.distributions import Independent, Normal

from tiny_dreamer_highway.models.decoder import (
    ContinuePredictor,
    ObservationDecoder,
    RewardPredictor,
)


# ── ObservationDecoder spatial math ──────────────────────────────────

def test_decoder_conv_transpose_spatial_formula() -> None:
    """ConvTranspose2d(k=4, s=2, p=1): out = (in-1)*stride - 2*pad + kernel.

    4 → (4-1)*2 - 2 + 4 = 8
    8 → 16 → 32 → 64
    """
    decoder = ObservationDecoder(latent_dim=40, output_shape=(1, 64, 64))

    # base_height/width = 64 // 16 = 4
    assert decoder.base_height == 4
    assert decoder.base_width == 4

    # Forward pass should produce (1, 64, 64) final spatial dims
    x = torch.randn(1, 40)
    out = decoder(x)
    assert out.shape == (1, 1, 64, 64)


def test_decoder_projection_dimensions() -> None:
    """Linear projects latent_dim → base_channels * base_h * base_w."""
    decoder = ObservationDecoder(latent_dim=230, output_shape=(1, 64, 64))

    expected_projection_out = 256 * 4 * 4  # 4096
    assert decoder.projection.in_features == 230
    assert decoder.projection.out_features == expected_projection_out


def test_decoder_non_square_output() -> None:
    """Non-square output: (1, 64, 32) → base 4×2."""
    decoder = ObservationDecoder(latent_dim=40, output_shape=(1, 64, 32))
    assert decoder.base_height == 4
    assert decoder.base_width == 2

    x = torch.randn(2, 40)
    out = decoder(x)
    assert out.shape == (2, 1, 64, 32)


def test_decoder_multichannel_output() -> None:
    """3-channel output shape."""
    decoder = ObservationDecoder(latent_dim=40, output_shape=(3, 64, 64))

    x = torch.randn(1, 40)
    out = decoder(x)
    assert out.shape == (1, 3, 64, 64)


# ── ObservationDecoder distribution ──────────────────────────────────

def test_decoder_distribution_uses_fixed_std() -> None:
    """distribution() returns Normal with std = distribution_std (constant, not learned)."""
    decoder = ObservationDecoder(latent_dim=40, output_shape=(1, 64, 64), distribution_std=2.5)

    x = torch.randn(1, 40)
    dist = decoder.distribution(x)

    # The std should be 2.5 everywhere
    assert isinstance(dist, Independent)
    base_dist = dist.base_dist
    assert isinstance(base_dist, Normal)
    assert torch.allclose(base_dist.scale, torch.full_like(base_dist.scale, 2.5))


def test_decoder_distribution_log_prob_shape() -> None:
    """log_prob shape = (B,) since event dims are reinterpreted."""
    decoder = ObservationDecoder(latent_dim=40, output_shape=(1, 64, 64), distribution_std=1.0)

    x = torch.randn(3, 40)
    dist = decoder.distribution(x)
    target = torch.randn(3, 1, 64, 64)
    lp = dist.log_prob(target)

    assert lp.shape == (3,)


# ── ObservationDecoder time dimension ────────────────────────────────

def test_decoder_handles_time_dimension() -> None:
    """(B, T, latent) → (B, T, C, H, W)."""
    decoder = ObservationDecoder(latent_dim=40, output_shape=(1, 64, 64))

    x = torch.randn(2, 5, 40)
    out = decoder(x)
    assert out.shape == (2, 5, 1, 64, 64)


# ── ObservationDecoder determinism ───────────────────────────────────

def test_decoder_seeded_forward_is_deterministic() -> None:
    x = torch.randn(2, 40)

    torch.manual_seed(42)
    d1 = ObservationDecoder(latent_dim=40, output_shape=(1, 64, 64))
    out1 = d1(x)

    torch.manual_seed(42)
    d2 = ObservationDecoder(latent_dim=40, output_shape=(1, 64, 64))
    out2 = d2(x)

    assert torch.equal(out1, out2)


# ── RewardPredictor ──────────────────────────────────────────────────

def test_reward_predictor_output_is_scalar_per_item() -> None:
    rp = RewardPredictor(latent_dim=40, hidden_dim=32, num_layers=2)

    x = torch.randn(4, 40)
    out = rp(x)
    assert out.shape == (4, 1)


def test_reward_predictor_distribution_uses_fixed_std() -> None:
    """Reward distribution std is fixed, not learned."""
    rp = RewardPredictor(latent_dim=40, hidden_dim=32, distribution_std=0.5)

    x = torch.randn(2, 40)
    dist = rp.distribution(x)
    base_dist = dist.base_dist
    assert torch.allclose(base_dist.scale, torch.full_like(base_dist.scale, 0.5))


def test_reward_predictor_distribution_log_prob_gaussian() -> None:
    """Verify log_prob matches hand-computed Gaussian NLL for known values.

    For μ=0.0, σ=1.0, x=0.0:
      log p = -0.5 * ln(2π) - ln(1) - 0 = -0.5*ln(2π) ≈ -0.9189
    """
    import math

    rp = RewardPredictor(latent_dim=40, hidden_dim=32, distribution_std=1.0)

    # Force the predictor to output exactly 0.0
    with torch.no_grad():
        for p in rp.parameters():
            p.zero_()

    x = torch.zeros(1, 40)
    dist = rp.distribution(x)
    target = torch.zeros(1, 1)
    lp = dist.log_prob(target)

    expected = -0.5 * math.log(2 * math.pi)
    assert torch.allclose(lp, torch.tensor([expected]), atol=1e-4)


def test_reward_predictor_handles_time_dim() -> None:
    rp = RewardPredictor(latent_dim=40, hidden_dim=32, num_layers=2)

    x = torch.randn(2, 5, 40)
    out = rp(x)
    assert out.shape == (2, 5, 1)


# ── ContinuePredictor ───────────────────────────────────────────────

def test_continue_predictor_output_is_logit() -> None:
    """Output is a raw logit (no sigmoid applied)."""
    cp = ContinuePredictor(latent_dim=40, hidden_dim=32, num_layers=2)

    # Zero all weights → output should be 0.0 (logit for P=0.5)
    with torch.no_grad():
        for p in cp.parameters():
            p.zero_()

    x = torch.zeros(1, 40)
    logit = cp(x)
    assert logit.shape == (1, 1)
    assert torch.allclose(logit, torch.zeros(1, 1), atol=1e-6)

    # σ(0) = 0.5
    prob = torch.sigmoid(logit)
    assert torch.allclose(prob, torch.tensor([[0.5]]), atol=1e-6)


def test_continue_predictor_handles_time_dim() -> None:
    cp = ContinuePredictor(latent_dim=40, hidden_dim=32, num_layers=2)

    x = torch.randn(2, 5, 40)
    out = cp(x)
    assert out.shape == (2, 5, 1)
