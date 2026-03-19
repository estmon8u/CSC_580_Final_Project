"""Deterministic mathematical tests for action processing utilities.

Verifies exact numerics: clamping boundaries, smoothing convergence,
lateral_enabled=False behaviour, and multi-step EMA convergence.
"""

import numpy as np
import torch

from tiny_dreamer_highway.utils.action_processing import (
    stabilize_action_array,
    stabilize_action_tensor,
)


# ── clamping boundary ────────────────────────────────────────────────

def test_tensor_clamping_after_scaling() -> None:
    """Actions exceeding [-1, 1] after scaling are clamped.

    action = [2.0, -2.0], longitudinal_scale=1.0, lateral_scale=1.0
    → scaled = [2.0, -2.0] → clamped = [1.0, -1.0]
    """
    action = torch.tensor([[2.0, -2.0]])
    result = stabilize_action_tensor(action)

    expected = torch.tensor([[1.0, -1.0]])
    assert torch.allclose(result, expected)


def test_array_clamping_after_scaling() -> None:
    action = np.array([2.0, -2.0], dtype=np.float32)
    result = stabilize_action_array(action)

    expected = np.array([1.0, -1.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_tensor_scaling_then_clamping() -> None:
    """longitudinal_scale=0.5, lateral_scale=0.3.

    action = [1.0, 1.0]
    → scaled = [0.5, 0.3] (no clamping needed)
    """
    action = torch.tensor([[1.0, 1.0]])
    result = stabilize_action_tensor(
        action, longitudinal_scale=0.5, lateral_scale=0.3,
    )

    expected = torch.tensor([[0.5, 0.3]])
    assert torch.allclose(result, expected)


def test_tensor_large_action_with_scaling_clamps() -> None:
    """action = [3.0, 5.0], scales = [1.0, 0.35].

    → scaled = [3.0, 1.75] → clamped = [1.0, 1.0]
    """
    action = torch.tensor([[3.0, 5.0]])
    result = stabilize_action_tensor(
        action, longitudinal_scale=1.0, lateral_scale=0.35,
    )

    expected = torch.tensor([[1.0, 1.0]])
    assert torch.allclose(result, expected)


# ── smoothing (EMA) ─────────────────────────────────────────────────

def test_tensor_smoothing_formula() -> None:
    """smoothed = factor * prev + (1-factor) * current.

    prev = [0.0, 0.0], current = [1.0, 1.0], factor = 0.6
    → smoothed = 0.6*[0,0] + 0.4*[1,1] = [0.4, 0.4]
    (with lateral_scale=1.0 to avoid extra scaling)
    """
    action = torch.tensor([[1.0, 1.0]])
    prev = torch.tensor([[0.0, 0.0]])
    result = stabilize_action_tensor(
        action, previous_action=prev,
        smoothing_factor=0.6, lateral_scale=1.0,
    )

    expected = torch.tensor([[0.4, 0.4]])
    assert torch.allclose(result, expected, atol=1e-6)


def test_array_smoothing_formula() -> None:
    action = np.array([1.0, 1.0], dtype=np.float32)
    prev = np.array([0.0, 0.0], dtype=np.float32)
    result = stabilize_action_array(
        action, previous_action=prev,
        smoothing_factor=0.6, lateral_scale=1.0,
    )

    expected = np.array([0.4, 0.4], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_smoothing_convergence() -> None:
    """Repeated smoothing with same target converges to that target.

    After many iterations of EMA(factor=0.6) with constant input [0.8, 0.5]:
    the output should converge to [0.8, 0.5] (within clamping bounds).
    """
    target = torch.tensor([[0.8, 0.5]])
    current = torch.tensor([[0.0, 0.0]])

    for _ in range(50):
        current = stabilize_action_tensor(
            target, previous_action=current,
            smoothing_factor=0.6, lateral_scale=1.0,
        )

    assert torch.allclose(current, target, atol=1e-4)


def test_smoothing_factor_zero_means_no_smoothing() -> None:
    """factor=0: output = current action (no blending with previous)."""
    action = torch.tensor([[0.8, 0.3]])
    prev = torch.tensor([[0.1, 0.1]])

    result = stabilize_action_tensor(
        action, previous_action=prev,
        smoothing_factor=0.0, lateral_scale=1.0,
    )

    # With factor=0, the smoothing branch is skipped
    assert torch.allclose(result, action)


def test_smoothing_factor_one_means_full_previous() -> None:
    """factor=1: output = previous action entirely."""
    action = torch.tensor([[0.8, 0.3]])
    prev = torch.tensor([[0.1, 0.1]])

    result = stabilize_action_tensor(
        action, previous_action=prev,
        smoothing_factor=1.0, lateral_scale=1.0,
    )

    assert torch.allclose(result, prev)


# ── lateral_enabled=False ────────────────────────────────────────────

def test_lateral_disabled_skips_lateral_scaling() -> None:
    """lateral_enabled=False → lateral component keeps raw scaling (no lateral_scale)."""
    action = torch.tensor([[0.5, 0.9]])

    result = stabilize_action_tensor(
        action, longitudinal_scale=1.0, lateral_scale=0.35,
        lateral_enabled=False,
    )

    # longitudinal scaled by 1.0, lateral NOT scaled (lateral_enabled=False)
    expected = torch.tensor([[0.5, 0.9]])
    assert torch.allclose(result, expected)


def test_lateral_enabled_applies_lateral_scale() -> None:
    action = torch.tensor([[0.5, 1.0]])

    result = stabilize_action_tensor(
        action, longitudinal_scale=1.0, lateral_scale=0.35,
        lateral_enabled=True,
    )

    expected = torch.tensor([[0.5, 0.35]])
    assert torch.allclose(result, expected)


# ── no previous action ──────────────────────────────────────────────

def test_no_previous_action_skips_smoothing() -> None:
    """When previous_action=None, no smoothing is applied regardless of factor."""
    action = torch.tensor([[0.7, 0.3]])

    result = stabilize_action_tensor(
        action, previous_action=None,
        smoothing_factor=0.9, lateral_scale=1.0,
    )

    assert torch.allclose(result, action)


# ── single-dim action ───────────────────────────────────────────────

def test_single_dim_action_only_scales_longitudinal() -> None:
    """1D action → only longitudinal scaling, no lateral scaling."""
    action = torch.tensor([[0.8]])

    result = stabilize_action_tensor(
        action, longitudinal_scale=0.5, lateral_scale=0.35,
    )

    expected = torch.tensor([[0.4]])
    assert torch.allclose(result, expected)


# ── array dtype preservation ─────────────────────────────────────────

def test_array_output_is_float32() -> None:
    action = np.array([0.5, 0.5], dtype=np.float64)
    result = stabilize_action_array(action)
    assert result.dtype == np.float32


def test_tensor_clone_does_not_modify_input() -> None:
    """stabilize_action_tensor should not mutate the input tensor."""
    action = torch.tensor([[0.5, 0.8]])
    original = action.clone()

    stabilize_action_tensor(action, longitudinal_scale=0.5, lateral_scale=0.1)

    assert torch.equal(action, original)
