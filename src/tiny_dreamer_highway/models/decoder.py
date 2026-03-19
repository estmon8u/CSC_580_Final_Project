"""Decoder and reward heads for the world model.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
from torch.distributions import Independent, Normal
from torch import Tensor, nn


class ObservationDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int, int] = (1, 64, 64),
        hidden_channels: tuple[int, int, int, int] = (256, 128, 64, 32),
        distribution_std: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if distribution_std <= 0:
            raise ValueError("distribution_std must be positive")

        out_channels, height, width = output_shape
        if min(out_channels, height, width) <= 0:
            raise ValueError("output_shape values must be positive")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("output height and width must be divisible by 16")

        self.output_shape = output_shape
        self.distribution_std = distribution_std
        self.base_height = height // 16
        self.base_width = width // 16
        self.base_channels = hidden_channels[0]

        self.projection = nn.Linear(
            latent_dim,
            self.base_channels * self.base_height * self.base_width,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[0], hidden_channels[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels[1], hidden_channels[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels[2], hidden_channels[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels[3], out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.register_buffer('_dtype_buf', torch.zeros(1), persistent=False)

    def distribution(self, latent_features: Tensor) -> Independent:
        mean = self.forward(latent_features)
        std = torch.full_like(mean, self.distribution_std)
        return Independent(Normal(mean, std), len(self.output_shape))

    def forward(self, latent_features: Tensor) -> Tensor:
        latent_features = latent_features.to(dtype=self._dtype_buf.dtype)

        # Flatten Time into Batch: (B, T, Latent_Dim) -> (B*T, Latent_Dim)
        batch_shape = latent_features.shape[:-1]
        flat_features = latent_features.reshape(-1, latent_features.shape[-1])

        projected = self.projection(flat_features)
        reshaped = projected.reshape(
            -1,
            self.base_channels,
            self.base_height,
            self.base_width,
        )
        decoded = self.decoder(reshaped)

        # Unflatten back to (B, T, C, H, W)
        return decoded.reshape(*batch_shape, *self.output_shape)


class RewardPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 200,
        num_layers: int = 2,
        distribution_std: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
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

        self.network = nn.Sequential(*layers)

    def distribution(self, latent_features: Tensor) -> Independent:
        mean = self.forward(latent_features)
        std = torch.full_like(mean, self.distribution_std)
        return Independent(Normal(mean, std), 1)

    def forward(self, latent_features: Tensor) -> Tensor:
        latent_features = latent_features.to(dtype=next(self.parameters()).dtype)
        return self.network(latent_features)


class ContinuePredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 200, num_layers: int = 2) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        layers: list[nn.Module] = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)
        self.register_buffer('_dtype_buf', torch.zeros(1), persistent=False)

    def forward(self, latent_features: Tensor) -> Tensor:
        latent_features = latent_features.to(dtype=self._dtype_buf.dtype)
        return self.network(latent_features)