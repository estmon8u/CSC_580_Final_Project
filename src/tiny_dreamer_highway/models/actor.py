"""Actor network for imagined control.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Actor(nn.Module):
    """Stochastic actor with tanh-normal distribution (DreamerV1 §3).

    Outputs a reparameterised sample squashed through *tanh* so values
    are always in ``(-1, 1)``.  During ``eval()`` mode the mean is
    returned (deterministic) for reproducible evaluation rollouts.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.action_dim = action_dim
        self.min_std = min_std

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )

    def forward(self, latent_features: Tensor) -> Tensor:
        latent_features = latent_features.to(dtype=next(self.parameters()).dtype)
        raw = self.net(latent_features)
        mean, raw_std = raw.split(self.action_dim, dim=-1)
        std = F.softplus(raw_std) + self.min_std

        if self.training:
            # Reparameterised sample — gradients flow through rsample
            sample = mean + std * torch.randn_like(mean)
        else:
            sample = mean

        return torch.tanh(sample)