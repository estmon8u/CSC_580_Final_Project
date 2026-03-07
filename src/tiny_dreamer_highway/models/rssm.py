"""Recurrent state-space model utilities.

Name: Esteban Montelongo
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
        deterministic_dim: int = 200,
        stochastic_dim: int = 30,
        hidden_dim: int = 200,
        min_std: float = 0.1,
        num_layers: int = 2,
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

        # Multi-layer prior and posterior networks
        self.prior_model = self._build_fc_network(
            deterministic_dim, hidden_dim, 2 * stochastic_dim, num_layers,
        )
        self.posterior_model = self._build_fc_network(
            deterministic_dim + embedding_dim, hidden_dim, 2 * stochastic_dim, num_layers,
        )

    @staticmethod
    def _build_fc_network(
        in_dim: int, hidden_dim: int, out_dim: int, num_layers: int,
    ) -> nn.Sequential:
        """Build a fully connected network with ``num_layers`` hidden layers."""
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LatentState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        _dt = self._dtype
        deterministic = torch.zeros(batch_size, self.deterministic_dim, device=device, dtype=_dt)
        stochastic = torch.zeros(batch_size, self.stochastic_dim, device=device, dtype=_dt)
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

        _dt = self._dtype
        stochastic = prev_state.stochastic.to(dtype=_dt)
        deterministic = prev_state.deterministic.to(dtype=_dt)
        action = action.to(dtype=_dt)
        gru_input = self.input_layer(torch.cat([stochastic, action], dim=-1))
        return self.gru(gru_input, deterministic)

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

    def imagine_rollout(self, start_state: LatentState, actions: Tensor) -> list[LatentState]:
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, T, action_dim)")
        if start_state.deterministic is None or start_state.stochastic is None:
            raise ValueError("start_state must contain deterministic and stochastic tensors")

        state = start_state
        rollout: list[LatentState] = []
        for step in range(actions.shape[1]):
            state = self.imagine_step(state, actions[:, step])
            rollout.append(state)
        return rollout

    def observe_step(self, prev_state: LatentState, action: Tensor, embedding: Tensor) -> LatentState:
        deterministic = self._next_deterministic(prev_state, action)
        embedding = embedding.to(dtype=self._dtype)
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