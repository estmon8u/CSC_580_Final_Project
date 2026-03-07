"""Tests for safe continuous-action stabilization.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import numpy as np
import torch

from tiny_dreamer_highway.config import ActionConfig
from tiny_dreamer_highway.utils import stabilize_action_array, stabilize_action_tensor


def test_action_config_has_safe_defaults() -> None:
    config = ActionConfig()
    assert config.longitudinal_scale == 1.0
    assert config.lateral_scale == 0.35
    assert config.smoothing_factor == 0.6


def test_stabilize_action_tensor_scales_lateral_component() -> None:
    action = torch.tensor([[0.8, 1.0]], dtype=torch.float32)
    stabilized = stabilize_action_tensor(action, lateral_scale=0.25)
    assert torch.allclose(stabilized, torch.tensor([[0.8, 0.25]], dtype=torch.float32))


def test_stabilize_action_tensor_applies_smoothing() -> None:
    action = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    previous = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    stabilized = stabilize_action_tensor(action, previous_action=previous, smoothing_factor=0.75)
    assert torch.allclose(stabilized, torch.tensor([[0.25, 0.25]], dtype=torch.float32))


def test_stabilize_action_array_scales_and_clips() -> None:
    action = np.array([1.5, 1.0], dtype=np.float32)
    stabilized = stabilize_action_array(action, longitudinal_scale=1.0, lateral_scale=0.35)
    np.testing.assert_allclose(stabilized, np.array([1.0, 0.35], dtype=np.float32))


def test_stabilize_action_array_preserves_single_dimension() -> None:
    action = np.array([0.5], dtype=np.float32)
    stabilized = stabilize_action_array(action, lateral_enabled=False, smoothing_factor=0.6)
    np.testing.assert_allclose(stabilized, np.array([0.5], dtype=np.float32))