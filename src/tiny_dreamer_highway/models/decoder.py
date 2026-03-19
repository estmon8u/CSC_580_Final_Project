"""Observation decoder and auxiliary prediction heads for the world model.

The decoder reconstructs pixel observations from the RSSM’s latent
feature vector ``[h_t ; s_t]``.  It mirrors the encoder’s architecture
in reverse: a linear projection expands the latent vector to a small
spatial feature map, then 4 transposed convolutions up-sample back to
the original image resolution.

Reconstruction path:  ``(B, [T,] latent_dim)``  →  linear  →  reshape  →  4× ConvTranspose2d  →  ``(B, [T,] C, H, W)``

The ``RewardPredictor`` and ``ContinuePredictor`` are small MLP heads
that also consume the latent features and output scalar predictions.

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
    """Transposed-CNN decoder that reconstructs images from latent features.

    The decoder models the observation as a diagonal Gaussian with a
    fixed standard deviation (``distribution_std``).  The ``forward``
    method returns the *mean* of this distribution; the ``distribution``
    method wraps it in a ``torch.distributions.Independent(Normal(...))``
    for log-probability computation.

    Spatial up-sampling: the latent vector is projected and reshaped to
    ``(base_channels, H//16, W//16)``, then 4 transposed convolutions
    each double the spatial resolution back to ``(C, H, W)``.

    Args:
        latent_dim:       Dimensionality of the input ``[h_t ; s_t]``.
        output_shape:     Target image shape ``(C, H, W)``.
        hidden_channels:  Channel counts for each deconv layer.
        distribution_std: Fixed std of the Gaussian observation model.
    """

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
        # The conv stack will up-sample from (H//16, W//16) to (H, W)
        self.base_height = height // 16
        self.base_width = width // 16
        self.base_channels = hidden_channels[0]

        # Project latent features to a small spatial feature map
        self.projection = nn.Linear(
            latent_dim,
            self.base_channels * self.base_height * self.base_width,
        )
        # 4× transposed-conv layers, each doubling spatial dims
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
        """Return a diagonal Gaussian distribution over the reconstructed image."""
        mean = self.forward(latent_features)
        std = torch.full_like(mean, self.distribution_std)
        return Independent(Normal(mean, std), len(self.output_shape))

    def forward(self, latent_features: Tensor) -> Tensor:
        """Decode latent features into reconstructed image means.

        Handles both ``(B, latent_dim)`` and ``(B, T, latent_dim)`` inputs
        by flattening the time dimension into batch, running the deconv
        stack, and unflattening.

        Returns:
            Reconstructed images, shape ``(B, [T,] C, H, W)``.
        """
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
    """MLP head that predicts scalar rewards from latent features.

    Models the reward as a Gaussian with a fixed standard deviation.
    The ``forward`` method returns the mean; ``distribution`` wraps
    it in a ``Normal`` for log-prob computation.

    Args:
        latent_dim:       Width of the input latent feature vector.
        hidden_dim:       Width of hidden layers.
        num_layers:       Number of hidden layers.
        distribution_std: Fixed Gaussian std for the reward model.
    """

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
    """MLP head that predicts episode continuation probability.

    Outputs raw logits (applied to ``sigmoid`` or ``BCE_with_logits``
    downstream).  The target is ``1 − terminated``: 1.0 if the episode
    continues, 0.0 on true termination (collision).

    Args:
        latent_dim:  Width of the input latent feature vector.
        hidden_dim:  Width of hidden layers.
        num_layers:  Number of hidden layers.
    """

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