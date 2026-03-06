"""Critic network for imagined value estimation.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from torch import Tensor, nn


class Critic(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.value = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent_features: Tensor) -> Tensor:
        return self.value(latent_features)