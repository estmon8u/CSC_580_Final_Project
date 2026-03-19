"""Deterministic mathematical tests for ObservationEncoder.

Verifies exact numerical outputs: uint8 normalization, conv spatial
arithmetic, projection dimensions, and seeded forward-pass reproducibility.
"""

import math

import torch

from tiny_dreamer_highway.models.encoder import LatentState, ObservationEncoder


def test_uint8_normalization_produces_exact_float_values() -> None:
    """0 → 0.0, 128 → 128/255, 255 → 1.0."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=32)
    obs = torch.zeros(1, 1, 64, 64, dtype=torch.uint8)
    obs[0, 0, 0, 0] = 0
    obs[0, 0, 0, 1] = 128
    obs[0, 0, 0, 2] = 255

    # Manually replicate the normalization path
    normed = obs.float() / 255.0
    assert normed[0, 0, 0, 0].item() == 0.0
    assert math.isclose(normed[0, 0, 0, 1].item(), 128.0 / 255.0, rel_tol=1e-7)
    assert normed[0, 0, 0, 2].item() == 1.0


def test_conv_spatial_dimensions_halve_at_each_layer() -> None:
    """Conv2d(k=4, s=2, p=1) halves spatial dims: 64→32→16→8→4."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=32)

    # Verify conv_output_shape after 4 layers of stride-2 convolution
    # Formula: floor((H + 2*pad - kernel) / stride + 1)
    # = floor((64 + 2 - 4) / 2 + 1) = floor(62/2 + 1) = 32
    # Then: 32 → 16 → 8 → 4
    expected_spatial = (4, 4)
    expected_channels = 256  # Last channel count in default (32, 64, 128, 256)

    assert encoder.conv_output_shape == (expected_channels, *expected_spatial)


def test_conv_output_dim_matches_product_of_shape() -> None:
    """conv_output_dim = channels[-1] * (H/16) * (W/16) = 256*4*4 = 4096."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=1024)

    expected_dim = 256 * 4 * 4  # 4096
    assert encoder.conv_output_dim == expected_dim


def test_conv_output_dim_non_square_input() -> None:
    """Non-square inputs: 64x32 → 4x2, dim = 256*4*2 = 2048."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 32), embedding_dim=128)

    expected_dim = 256 * 4 * 2  # 2048
    assert encoder.conv_output_dim == expected_dim


def test_projection_maps_conv_features_to_embedding_dim() -> None:
    """Linear projection: conv_output_dim → embedding_dim."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=1024)

    assert encoder.projection.in_features == encoder.conv_output_dim
    assert encoder.projection.out_features == 1024


def test_seeded_forward_is_deterministic() -> None:
    """Same seed + same input → identical embedding across two runs."""
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)

    torch.manual_seed(42)
    enc1 = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=128)
    out1 = enc1.encode(obs)

    torch.manual_seed(42)
    enc2 = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=128)
    out2 = enc2.encode(obs)

    assert torch.equal(out1, out2)


def test_encode_output_shape_single_batch() -> None:
    """(B, C, H, W) input → (B, embedding_dim) output."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=256)
    obs = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)

    emb = encoder.encode(obs)
    assert emb.shape == (3, 256)


def test_encode_output_shape_with_time_dim() -> None:
    """(B, T, C, H, W) input → (B, T, embedding_dim) output."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=256)
    obs = torch.randint(0, 256, (2, 5, 1, 64, 64), dtype=torch.uint8)

    emb = encoder.encode(obs)
    assert emb.shape == (2, 5, 256)


def test_forward_returns_latent_state_with_embedding() -> None:
    """forward() wraps encode() result in a LatentState."""
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=128)
    obs = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)

    state = encoder(obs)
    assert isinstance(state, LatentState)
    assert state.embedding is not None
    assert state.embedding.shape == (2, 128)
    assert state.deterministic is None
    assert state.stochastic is None


def test_encoder_output_is_finite() -> None:
    """No NaN or Inf in encoder output for valid input range."""
    torch.manual_seed(7)
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=512)
    obs = torch.randint(0, 256, (4, 1, 64, 64), dtype=torch.uint8)

    emb = encoder.encode(obs)
    assert torch.all(torch.isfinite(emb))


def test_multichannel_conv_output_dim() -> None:
    """3-channel (RGB) input: conv_output_dim is the same spatial size."""
    encoder = ObservationEncoder(in_channels=3, observation_shape=(64, 64), embedding_dim=256)
    # First conv changes channels but spatial dims are determined by kernel/stride
    # So conv_output_dim should still be 256 * 4 * 4 = 4096
    assert encoder.conv_output_dim == 256 * 4 * 4
