"""Safe continuous-action post-processing helpers.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def stabilize_action_tensor(
    action: Tensor,
    previous_action: Tensor | None = None,
    *,
    longitudinal_scale: float = 1.0,
    lateral_scale: float = 1.0,
    smoothing_factor: float = 0.0,
    lateral_enabled: bool = True,
) -> Tensor:
    """Scale and smooth a continuous action tensor while preserving shape."""
    processed = action.clone()
    processed[..., 0] = processed[..., 0] * longitudinal_scale
    if lateral_enabled and processed.shape[-1] >= 2:
        processed[..., -1] = processed[..., -1] * lateral_scale
    if previous_action is not None and smoothing_factor > 0.0:
        processed = smoothing_factor * previous_action + (1.0 - smoothing_factor) * processed
    return processed.clamp(-1.0, 1.0)


def stabilize_action_array(
    action: np.ndarray,
    previous_action: np.ndarray | None = None,
    *,
    longitudinal_scale: float = 1.0,
    lateral_scale: float = 1.0,
    smoothing_factor: float = 0.0,
    lateral_enabled: bool = True,
) -> np.ndarray:
    """Scale and smooth a continuous action array while preserving shape."""
    processed = np.array(action, dtype=np.float32, copy=True)
    processed[..., 0] *= longitudinal_scale
    if lateral_enabled and processed.shape[-1] >= 2:
        processed[..., -1] *= lateral_scale
    if previous_action is not None and smoothing_factor > 0.0:
        previous = np.asarray(previous_action, dtype=np.float32)
        processed = smoothing_factor * previous + (1.0 - smoothing_factor) * processed
    return np.clip(processed, -1.0, 1.0).astype(np.float32, copy=False)