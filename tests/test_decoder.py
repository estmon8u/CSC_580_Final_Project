import torch

from tiny_dreamer_highway.models import ObservationDecoder, RewardPredictor


def test_observation_decoder_reconstructs_expected_shape() -> None:
    decoder = ObservationDecoder(latent_dim=160, output_shape=(1, 64, 64))
    latent_features = torch.randn(4, 160)

    reconstruction = decoder(latent_features)

    assert reconstruction.shape == (4, 1, 64, 64)


def test_observation_decoder_supports_multi_channel_outputs() -> None:
    decoder = ObservationDecoder(latent_dim=96, output_shape=(3, 64, 64))
    latent_features = torch.randn(2, 96)

    reconstruction = decoder(latent_features)

    assert reconstruction.shape == (2, 3, 64, 64)


def test_reward_predictor_returns_scalar_reward_per_batch_item() -> None:
    predictor = RewardPredictor(latent_dim=160, hidden_dim=64)
    latent_features = torch.randn(5, 160)

    rewards = predictor(latent_features)

    assert rewards.shape == (5, 1)