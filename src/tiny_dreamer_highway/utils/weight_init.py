"""Weight initialization utilities following the DreamerV1 reference.

Applies Kaiming uniform initialization to all Conv2d, ConvTranspose2d,
and Linear layers, with biases zeroed.  This is applied once during
model construction in ``TinyWorldModel``, ``Actor``, ``Critic``, and
``DiscreteActor``.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from torch import nn


def apply_kaiming_init(module: nn.Module) -> None:
    """Apply Kaiming uniform initialization to Conv2d, ConvTranspose2d, and Linear layers.

    Biases are zeroed.  This matches the reference Dreamer implementation.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # Use nonlinearity="leaky_relu" which is the closest Kaiming
            # mode to ELU (the dominant activation in our RSSM, actor, and
            # critic networks).  The negative slope default (0.01) is a
            # reasonable proxy for ELU's alpha=1.0.
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
