"""CNN observation encoder and LatentState dataclass for the world model.

The encoder maps pixel observations into compact embedding vectors that
the RSSM conditions on.  It uses a 4-layer convolutional stack with
stride-2 down-sampling followed by a linear projection to
``embedding_dim``.

Input path:  ``(B, [T,] C, H, W)``  →  conv stack  →  flatten  →  linear  →  ``(B, [T,] embedding_dim)``

The ``LatentState`` dataclass carries all components of the RSSM’s
recurrent state in a single object.

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
    """Container for the RSSM’s two-part latent state.

    Fields:
        embedding:     CNN encoder output ``e_t``, shape ``(B, embedding_dim)``.
        deterministic: GRU hidden state ``h_t``, shape ``(B, deterministic_dim)``.
        stochastic:    Sampled stochastic state ``s_t``, shape ``(B, stochastic_dim)``.
                       For categoricals: flattened one-hot ``(B, num_cat * num_classes)``.
        logits:        Raw categorical logits, shape ``(B, num_cat, num_classes)``.
                       ``None`` for Gaussian mode (V1 legacy).
        dist_mean:     Mean of the Gaussian distribution over ``s_t`` (V1 legacy).
        dist_std:      Std of the Gaussian distribution over ``s_t`` (V1 legacy).

    The ``features`` property returns the concatenated latent feature
    vector ``[s_t ; h_t]`` used by all downstream heads (decoder,
    reward, continue, actor, critic).
    """

    embedding: Tensor | None = None
    deterministic: Tensor | None = None
    stochastic: Tensor | None = None
    logits: Tensor | None = None
    dist_mean: Tensor | None = None
    dist_std: Tensor | None = None

    @property
    def features(self) -> Tensor:
        """Return the latent feature vector ``[s_t ; h_t]``.

        Falls back to the raw embedding if no RSSM state is present
        (e.g., in the encoder-only forward path).
        """
        parts = [part for part in (self.stochastic, self.deterministic) if part is not None]
        if parts:
            if len(parts) == 1:
                return parts[0]
            return torch.cat(parts, dim=-1)
        if self.embedding is not None:
            return self.embedding
        raise ValueError("LatentState must contain at least one tensor")


class ObservationEncoder(nn.Module):
    """4-layer CNN encoder that maps pixel observations to embedding vectors.

    Architecture: 4 × (Conv2d stride-2 + ReLU) → flatten → Linear.
    Each conv layer halves the spatial dimensions, so a 64×64 input
    becomes 4×4 after 4 layers.  The flattened output is projected
    to ``embedding_dim`` by a linear layer.

    Supports batched ``(B, C, H, W)`` and sequence ``(B, T, C, H, W)``
    inputs — the time dimension is folded into the batch for efficient
    convolution and unfolded afterward.

    Args:
        in_channels:       Number of input channels (1 for grayscale).
        observation_shape: Spatial dimensions ``(H, W)``.
        channels:          Output channels for each conv layer.
        embedding_dim:     Dimensionality of the output embedding.
    """

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

        # Probe the conv stack with a dummy tensor to compute the exact
        # flattened output size (avoids fragile manual calculation).
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *observation_shape, dtype=torch.float32)
            conv_output = self.conv_stack(dummy)
        self.conv_output_shape = tuple(conv_output.shape[1:])  # (C_out, H_out, W_out)
        self.conv_output_dim = int(conv_output.reshape(1, -1).shape[-1])
        self.projection = nn.Linear(self.conv_output_dim, embedding_dim)

        # Lightweight buffer to track the module’s current dtype
        self.register_buffer('_dtype_buf', torch.zeros(1), persistent=False)

    def encode(self, observations: Tensor) -> Tensor:
        """Encode observations into embedding vectors.

        Accepts 3-D ``(C, H, W)``, 4-D ``(B, C, H, W)``, or 5-D
        ``(B, T, C, H, W)`` inputs.  Automatically handles uint8→float
        conversion and batch/time flattening.

        Returns:
            Embedding tensor with the batch/time prefix preserved:
            ``(B, embedding_dim)`` or ``(B, T, embedding_dim)``.
        """
        if observations.ndim not in (3, 4, 5):
            raise ValueError("observations must have shape (B, T, C, H, W), (B, C, H, W) or (C, H, W)")
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)

        # Flatten Time into Batch: (B, T, C, H, W) -> (B*T, C, H, W)
        batch_shape = observations.shape[:-3]
        flat_obs = observations.reshape(-1, *observations.shape[-3:])

        _dtype = self._dtype_buf.dtype
        features = flat_obs.to(dtype=_dtype)
        if observations.dtype == torch.uint8:
            features = features / 255.0

        encoded = self.conv_stack(features)
        flattened = encoded.reshape(encoded.shape[0], -1)
        projected = self.projection(flattened)

        # Unflatten back to (B, T, Embedding_Dim)
        return projected.reshape(*batch_shape, self.embedding_dim)

    def forward(self, observations: Tensor) -> LatentState:
        """Convenience forward: encode and wrap in a ``LatentState``."""
        return LatentState(embedding=self.encode(observations))