"""Weight initialization utilities (DreamerV1 reference pattern).

Name: Esteban
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
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
