"""Critic network for imagined value estimation.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from torch import Tensor, nn

from tiny_dreamer_highway.utils.weight_init import apply_kaiming_init


class Critic(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 200,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        layers: list[nn.Module] = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))

        self.value = nn.Sequential(*layers)

        apply_kaiming_init(self)

    def forward(self, latent_features: Tensor) -> Tensor:
        latent_features = latent_features.to(dtype=next(self.parameters()).dtype)
        return self.value(latent_features)