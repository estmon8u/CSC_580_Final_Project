"""Combined world-model forward pass utilities.

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
    embedding: Tensor
    prior_state: LatentState
    posterior_state: LatentState
    reconstruction: Tensor
    predicted_reward: Tensor
    predicted_continue: Tensor | None = None


class TinyWorldModel(nn.Module):
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
        reward_hidden_dim: int = 200,
        reward_num_layers: int = 2,
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
        self.decoder = ObservationDecoder(latent_dim=latent_dim, output_shape=observation_shape)
        self.reward_predictor = RewardPredictor(
            latent_dim=latent_dim,
            hidden_dim=reward_hidden_dim,
            num_layers=reward_num_layers,
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
            predicted_reward=predicted_reward,
            predicted_continue=predicted_continue,
        )