"""Discrete actor network for DreamerV1 behavior learning.

This actor handles discrete action spaces (e.g., Highway-Env’s
``DiscreteMetaAction`` for lane changes and speed adjustments).
It outputs one-hot action vectors using temperature-controlled
Gumbel-Softmax during training for differentiable action selection,
and hard argmax during evaluation for deterministic behavior.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tiny_dreamer_highway.utils.weight_init import apply_kaiming_init


class DiscreteActor(nn.Module):
    """Categorical actor for discrete action spaces.

    During ``training`` mode, uses Gumbel-Softmax (straight-through) so
    that the one-hot output is differentiable through the sampling
    operation, enabling imagination-based policy learning.  During
    ``eval()`` mode, returns a hard one-hot from argmax for deterministic
    action selection.

    Args:
        latent_dim:          Width of the input latent feature vector.
        num_actions:         Number of discrete actions.
        hidden_dim:          Width of each hidden layer.
        num_layers:          Number of hidden layers.
        gumbel_temperature:  Temperature for Gumbel-Softmax (lower = harder).
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 200,
        num_layers: int = 2,
        gumbel_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        self.num_actions = num_actions
        self.gumbel_temperature = gumbel_temperature

        layers: list[nn.Module] = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_actions))

        self.net = nn.Sequential(*layers)
        self.register_buffer('_dtype_buf', torch.zeros(1), persistent=False)

        apply_kaiming_init(self)

    def forward(self, latent_features: Tensor) -> Tensor:
        """Return a one-hot action tensor.

        Training: Gumbel-Softmax straight-through (differentiable).
        Eval: Hard one-hot from argmax (deterministic).
        """
        latent_features = latent_features.to(dtype=self._dtype_buf.dtype)
        logits = self.net(latent_features)

        if self.training:
            # Straight-through Gumbel-Softmax: forward is hard one-hot,
            # backward uses the soft gradient
            return F.gumbel_softmax(
                logits, tau=self.gumbel_temperature, hard=True, dim=-1
            )
        else:
            # Deterministic argmax → hard one-hot
            indices = logits.argmax(dim=-1)
            return F.one_hot(indices, self.num_actions).to(logits.dtype)
