"""Continuous actor network for DreamerV1 behavior learning.

The actor maps latent features ``[h_t ; s_t]`` to continuous actions
via a tanh-normal distribution.  During training, actions are sampled
with the reparameterization trick (tanh transform applied to a normal
sample) so gradients flow through the sampling.  During evaluation,
only the mean is used for deterministic action selection.

The ``init_std`` / ``mean_scale`` parameterization follows the
reference Dreamer implementation to ensure wide initial exploration.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from tiny_dreamer_highway.utils.weight_init import apply_kaiming_init


class Actor(nn.Module):
    """Stochastic actor with tanh-normal distribution (DreamerV1 §3).

    Uses ``init_std`` / ``mean_scale`` parameterisation from the reference
    Dreamer implementation for wider initial exploration.  During
    ``eval()`` mode the *mean* of the tanh-normal is returned for
    reproducible evaluation rollouts.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 200,
        num_layers: int = 2,
        init_std: float = 5.0,
        mean_scale: float = 5.0,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.action_dim = action_dim
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.min_std = min_std

        layers: list[nn.Module] = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 2 * action_dim))

        self.net = nn.Sequential(*layers)
        self.register_buffer('_dtype_buf', torch.zeros(1), persistent=False)

        apply_kaiming_init(self)

    def forward(self, latent_features: Tensor) -> Tensor:
        latent_features = latent_features.to(dtype=self._dtype_buf.dtype)
        raw = self.net(latent_features)
        mean, raw_std = raw.split(self.action_dim, dim=-1)

        # Scale mean through tanh to bound it, then rescale
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)

        # Wide initial std via init_std offset (softplus(raw + init_std))
        std = F.softplus(raw_std + self.init_std) + self.min_std

        if self.training:
            # Reparameterised tanh-normal sample with Jacobian correction
            dist = Normal(mean, std)
            dist = Independent(dist, 1)
            dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
            return dist.rsample()
        else:
            return torch.tanh(mean)