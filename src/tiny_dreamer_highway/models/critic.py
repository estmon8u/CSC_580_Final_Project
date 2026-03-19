"""Critic (value) network for DreamerV1 behavior learning.

The critic estimates the expected discounted return from a given latent
state.  During behavior learning, the actor proposes actions, the RSSM
imagines future states via the prior, and the critic evaluates the
resulting trajectory to produce λ-return targets.

The value is modeled as a Gaussian with a fixed standard deviation,
following the same probabilistic convention as the decoder and reward
heads.  The ``forward`` method returns the scalar mean prediction;
``distribution`` wraps it in a ``Normal`` for optional log-prob usage.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
from torch.distributions import Independent, Normal
from torch import Tensor, nn

from tiny_dreamer_highway.utils.weight_init import apply_kaiming_init


class Critic(nn.Module):
    """MLP value function: latent features → scalar state value.

    Architecture: ``num_layers`` × (Linear + ELU) → Linear(1).
    Initialized with Kaiming uniform for stable early training.

    Args:
        latent_dim:       Width of the input ``[h_t ; s_t]`` vector.
        hidden_dim:       Width of each hidden layer.
        num_layers:       Number of hidden layers.
        distribution_std: Fixed std for the Gaussian value model.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 200,
        num_layers: int = 3,
        distribution_std: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if distribution_std <= 0:
            raise ValueError("distribution_std must be positive")

        self.distribution_std = distribution_std

        layers: list[nn.Module] = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))

        self.value = nn.Sequential(*layers)
        self.register_buffer('_dtype_buf', torch.zeros(1), persistent=False)

        apply_kaiming_init(self)

    def distribution(self, latent_features: Tensor) -> Independent:
        mean = self.forward(latent_features)
        std = torch.full_like(mean, self.distribution_std)
        return Independent(Normal(mean, std), 1)

    def forward(self, latent_features: Tensor) -> Tensor:
        latent_features = latent_features.to(dtype=self._dtype_buf.dtype)
        return self.value(latent_features)