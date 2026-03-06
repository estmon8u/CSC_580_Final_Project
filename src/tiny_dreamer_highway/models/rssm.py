"""Recurrent state-space model utilities.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tiny_dreamer_highway.models.encoder import LatentState


class RecurrentStateSpaceModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embedding_dim: int,
        deterministic_dim: int = 128,
        stochastic_dim: int = 32,
        hidden_dim: int = 128,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if deterministic_dim <= 0:
            raise ValueError("deterministic_dim must be positive")
        if stochastic_dim <= 0:
            raise ValueError("stochastic_dim must be positive")

        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        self.input_layer = nn.Sequential(
            nn.Linear(action_dim + stochastic_dim, hidden_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deterministic_dim)

        self.prior_model = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim),
        )
        self.posterior_model = nn.Sequential(
            nn.Linear(deterministic_dim + embedding_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LatentState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        deterministic = torch.zeros(batch_size, self.deterministic_dim, device=device)
        stochastic = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return LatentState(deterministic=deterministic, stochastic=stochastic)

    def _distribution_parameters(self, stats: Tensor) -> tuple[Tensor, Tensor]:
        mean, std_param = torch.chunk(stats, 2, dim=-1)
        std = torch.nn.functional.softplus(std_param) + self.min_std
        return mean, std

    def _sample_stochastic(self, mean: Tensor, std: Tensor) -> Tensor:
        noise = torch.randn_like(mean)
        return mean + std * noise

    def _next_deterministic(self, prev_state: LatentState, action: Tensor) -> Tensor:
        if prev_state.stochastic is None or prev_state.deterministic is None:
            raise ValueError("prev_state must contain stochastic and deterministic tensors")

        gru_input = self.input_layer(torch.cat([prev_state.stochastic, action], dim=-1))
        return self.gru(gru_input, prev_state.deterministic)

    def imagine_step(self, prev_state: LatentState, action: Tensor) -> LatentState:
        deterministic = self._next_deterministic(prev_state, action)
        prior_stats = self.prior_model(deterministic)
        prior_mean, prior_std = self._distribution_parameters(prior_stats)
        stochastic = self._sample_stochastic(prior_mean, prior_std)
        return LatentState(
            deterministic=deterministic,
            stochastic=stochastic,
            dist_mean=prior_mean,
            dist_std=prior_std,
        )

    def observe_step(self, prev_state: LatentState, action: Tensor, embedding: Tensor) -> LatentState:
        deterministic = self._next_deterministic(prev_state, action)
        posterior_stats = self.posterior_model(torch.cat([deterministic, embedding], dim=-1))
        posterior_mean, posterior_std = self._distribution_parameters(posterior_stats)
        stochastic = self._sample_stochastic(posterior_mean, posterior_std)
        return LatentState(
            embedding=embedding,
            deterministic=deterministic,
            stochastic=stochastic,
            dist_mean=posterior_mean,
            dist_std=posterior_std,
        )