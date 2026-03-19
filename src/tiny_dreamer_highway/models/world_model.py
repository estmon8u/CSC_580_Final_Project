"""Combined world model: encoder + RSSM + decoder + prediction heads.

The ``TinyWorldModel`` bundles all components of the DreamerV1 world
model into a single ``nn.Module``:

* ``ObservationEncoder`` (CNN) — maps pixels to embeddings.
* ``RecurrentStateSpaceModel`` (RSSM) — maintains latent dynamics.
* ``ObservationDecoder`` (transposed CNN) — reconstructs pixels.
* ``RewardPredictor`` (MLP) — predicts scalar rewards.
* ``ContinuePredictor`` (MLP, optional) — predicts episode continuation.

The single-step ``forward`` method is used during policy collection
to update the latent state from a new observation.  The sequence-level
training path in ``sequence_world_model_step.py`` calls the sub-modules
directly for efficiency.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tiny_dreamer_highway.models.decoder import ContinuePredictor, ObservationDecoder, RewardPredictor
from tiny_dreamer_highway.models.encoder import LatentState, ObservationEncoder
from tiny_dreamer_highway.models.rssm import RecurrentStateSpaceModel
from tiny_dreamer_highway.utils.weight_init import apply_kaiming_init


@dataclass(slots=True)
class WorldModelOutput:
    """Output bundle from a single-step world-model forward pass.

    Contains the observation embedding, prior and posterior latent
    states, the reconstructed image, and all head predictions.
    """

    embedding: Tensor
    prior_state: LatentState
    posterior_state: LatentState
    reconstruction: Tensor
    predicted_reward: Tensor
    predicted_observation_std: float | None = None
    predicted_reward_std: float | None = None
    predicted_continue: Tensor | None = None


class TinyWorldModel(nn.Module):
    """Full DreamerV1 world model combining all sub-modules.

    During construction, Kaiming uniform initialization is applied to
    all Conv2d and Linear layers for stable training.

    Args:
        observation_shape:           ``(C, H, W)`` of the input images.
        action_dim:                  Number of action dimensions.
        embedding_dim:               Encoder output width.
        deterministic_dim:           GRU hidden state width.
        stochastic_dim:              Stochastic state width.
        hidden_dim:                  RSSM MLP hidden layer width.
        rssm_min_std:                Minimum std for the RSSM Gaussians.
        rssm_num_layers:             Hidden layers in RSSM prior/posterior.
        observation_distribution_std: Fixed std for decoder Gaussian.
        reward_hidden_dim:           Reward head hidden width.
        reward_num_layers:           Reward head hidden layers.
        reward_distribution_std:     Fixed std for reward Gaussian.
        use_continue_model:          Whether to include a continue head.
        continue_hidden_dim:         Continue head hidden width.
        continue_num_layers:         Continue head hidden layers.
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int] = (1, 64, 64),
        action_dim: int = 2,
        embedding_dim: int = 1024,
        deterministic_dim: int = 200,
        stochastic_dim: int = 30,
        hidden_dim: int = 200,
        rssm_min_std: float = 0.1,
        rssm_num_layers: int = 2,
        observation_distribution_std: float = 1.0,
        reward_hidden_dim: int = 200,
        reward_num_layers: int = 2,
        reward_distribution_std: float = 1.0,
        use_continue_model: bool = True,
        continue_hidden_dim: int = 200,
        continue_num_layers: int = 2,
    ) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.encoder = ObservationEncoder(
            in_channels=channels,
            observation_shape=(height, width),
            embedding_dim=embedding_dim,
        )
        self.rssm = RecurrentStateSpaceModel(
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            deterministic_dim=deterministic_dim,
            stochastic_dim=stochastic_dim,
            hidden_dim=hidden_dim,
            min_std=rssm_min_std,
            num_layers=rssm_num_layers,
        )
        latent_dim = deterministic_dim + stochastic_dim
        self.decoder = ObservationDecoder(
            latent_dim=latent_dim,
            output_shape=observation_shape,
            distribution_std=observation_distribution_std,
        )
        self.reward_predictor = RewardPredictor(
            latent_dim=latent_dim,
            hidden_dim=reward_hidden_dim,
            num_layers=reward_num_layers,
            distribution_std=reward_distribution_std,
        )
        self.continue_predictor = (
            ContinuePredictor(
                latent_dim=latent_dim,
                hidden_dim=continue_hidden_dim,
                num_layers=continue_num_layers,
            )
            if use_continue_model
            else None
        )

        # Kaiming uniform initialization for all Conv/Linear layers
        apply_kaiming_init(self)

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        prev_state: LatentState | None = None,
    ) -> WorldModelOutput:
        """Single-step forward pass used during policy collection.

        Encodes the observation, advances the RSSM (computing both prior
        and posterior), and runs the decoder + prediction heads on the
        posterior features.

        Args:
            observations: Current frame, shape ``(B, C, H, W)`` or ``(C, H, W)``.
            actions:      Action taken, shape ``(B, action_dim)`` or ``(action_dim,)``.
            prev_state:   Previous RSSM state (defaults to zeros).

        Returns:
            ``WorldModelOutput`` with embeddings, latent states, and predictions.
        """
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("observations and actions must have matching batch dimensions")

        embedding = self.encoder.encode(observations)
        if prev_state is None:
            prev_state = self.rssm.initial_state(batch_size=observations.shape[0], device=observations.device)

        prior_state = self.rssm.imagine_step(prev_state, actions)
        posterior_state = self.rssm.observe_step(prev_state, actions, embedding)
        latent_features = posterior_state.features
        reconstruction = self.decoder(latent_features)
        predicted_reward = self.reward_predictor(latent_features)
        predicted_continue = (
            self.continue_predictor(latent_features) if self.continue_predictor is not None else None
        )
        return WorldModelOutput(
            embedding=embedding,
            prior_state=prior_state,
            posterior_state=posterior_state,
            reconstruction=reconstruction,
            predicted_observation_std=self.decoder.distribution_std,
            predicted_reward=predicted_reward,
            predicted_reward_std=self.reward_predictor.distribution_std,
            predicted_continue=predicted_continue,
        )