import pytest
import torch

from tiny_dreamer_highway.models import LatentState, ObservationEncoder


def test_observation_encoder_returns_expected_embedding_shape() -> None:
    encoder = ObservationEncoder(in_channels=1, observation_shape=(64, 64), embedding_dim=256)
    observations = torch.randint(0, 256, (4, 1, 64, 64), dtype=torch.uint8)

    latent = encoder(observations)

    assert latent.embedding is not None
    assert latent.embedding.shape == (4, 256)
    assert latent.features.shape == (4, 256)


def test_observation_encoder_supports_frame_stacks() -> None:
    encoder = ObservationEncoder(in_channels=3, observation_shape=(64, 64), embedding_dim=128)
    observations = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)

    latent = encoder(observations)

    assert latent.embedding is not None
    assert latent.embedding.shape == (2, 128)


def test_latent_state_concatenates_stochastic_and_deterministic_features() -> None:
    latent = LatentState(
        deterministic=torch.zeros(3, 32),
        stochastic=torch.ones(3, 16),
    )

    assert latent.features.shape == (3, 48)


def test_latent_state_requires_at_least_one_tensor() -> None:
    latent = LatentState()

    with pytest.raises(ValueError, match="at least one tensor"):
        _ = latent.features


def test_latent_state_prefers_world_model_features_over_embedding() -> None:
    latent = LatentState(
        embedding=torch.randn(2, 256),
        deterministic=torch.zeros(2, 32),
        stochastic=torch.ones(2, 16),
    )

    assert latent.features.shape == (2, 48)