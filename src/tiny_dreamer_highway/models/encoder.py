"""World-model encoder utilities.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class LatentState:
    embedding: Tensor | None = None
    deterministic: Tensor | None = None
    stochastic: Tensor | None = None
    dist_mean: Tensor | None = None
    dist_std: Tensor | None = None

    @property
    def features(self) -> Tensor:
        parts = [part for part in (self.stochastic, self.deterministic) if part is not None]
        if parts:
            if len(parts) == 1:
                return parts[0]
            return torch.cat(parts, dim=-1)
        if self.embedding is not None:
            return self.embedding
        raise ValueError("LatentState must contain at least one tensor")


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        observation_shape: tuple[int, int] = (64, 64),
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)
        self.observation_shape = observation_shape
        self.embedding_dim = embedding_dim

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *observation_shape, dtype=torch.float32)
            conv_output = self.conv_stack(dummy)
        self.conv_output_shape = tuple(conv_output.shape[1:])
        self.conv_output_dim = int(conv_output.reshape(1, -1).shape[-1])
        self.projection = nn.Linear(self.conv_output_dim, embedding_dim)

    def encode(self, observations: Tensor) -> Tensor:
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        if observations.ndim != 4:
            raise ValueError("observations must have shape (B, C, H, W) or (C, H, W)")

        # Cast to the conv stack's own dtype (fp32 normally, bf16 under AMP/Flash).
        _dtype = next(self.conv_stack.parameters()).dtype
        features = observations.to(dtype=_dtype)
        if observations.dtype == torch.uint8:
            features = features / 255.0

        encoded = self.conv_stack(features)
        flattened = encoded.reshape(encoded.shape[0], -1)
        return self.projection(flattened)

    def forward(self, observations: Tensor) -> LatentState:
        return LatentState(embedding=self.encode(observations))