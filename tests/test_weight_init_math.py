"""Deterministic mathematical tests for Kaiming weight initialization.

Verifies: weight variance ≈ 2/fan_in (Kaiming uniform), all biases
exactly zero, and correct application to different layer types.
"""

import math

import torch
from torch import nn

from tiny_dreamer_highway.utils.weight_init import apply_kaiming_init


def test_biases_are_exactly_zero() -> None:
    """All biases must be exactly 0 after apply_kaiming_init."""
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.Linear(200, 50),
        nn.Conv2d(3, 16, kernel_size=3),
    )
    apply_kaiming_init(model)

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            assert torch.equal(m.bias.data, torch.zeros_like(m.bias.data))


def test_linear_weight_variance_kaiming_uniform() -> None:
    """Kaiming uniform: weights ~ U(-bound, bound) where bound = √(3 * 2/fan_in).

    Var(U(-a, a)) = a²/3
    bound = √(6 / fan_in)  [since gain=√(2/(1+a²)) with a=0.01]
    Var ≈ (6/fan_in) / 3 = 2/fan_in

    Allow ±30% tolerance for statistical sampling.
    """
    torch.manual_seed(42)
    layer = nn.Linear(1000, 500)  # large for stable statistics
    apply_kaiming_init(nn.Sequential(layer))

    fan_in = 1000
    expected_var = 2.0 / fan_in  # 0.002
    actual_var = layer.weight.data.var().item()

    # Allow generous tolerance for random variance
    assert abs(actual_var - expected_var) / expected_var < 0.3, (
        f"Expected var ≈ {expected_var:.4f}, got {actual_var:.4f}"
    )


def test_conv2d_weight_variance_kaiming_uniform() -> None:
    """Conv2d: fan_in = in_channels * kernel_h * kernel_w."""
    torch.manual_seed(42)
    layer = nn.Conv2d(64, 128, kernel_size=4)  # fan_in = 64 * 4 * 4 = 1024
    apply_kaiming_init(nn.Sequential(layer))

    fan_in = 64 * 4 * 4  # 1024
    expected_var = 2.0 / fan_in
    actual_var = layer.weight.data.var().item()

    assert abs(actual_var - expected_var) / expected_var < 0.3


def test_conv_transpose2d_weight_variance() -> None:
    """ConvTranspose2d also uses Kaiming init."""
    torch.manual_seed(42)
    layer = nn.ConvTranspose2d(128, 64, kernel_size=4)
    apply_kaiming_init(nn.Sequential(layer))

    # For ConvTranspose2d, PyTorch Kaiming uses fan_in = out_channels * kH * kW
    # (since transposed conv has reversed semantics)
    fan_in = 64 * 4 * 4  # 1024
    expected_var = 2.0 / fan_in
    actual_var = layer.weight.data.var().item()

    assert abs(actual_var - expected_var) / expected_var < 0.3


def test_init_only_affects_matching_layers() -> None:
    """BatchNorm layers are NOT modified by apply_kaiming_init."""
    bn = nn.BatchNorm2d(16)
    original_weight = bn.weight.data.clone()
    original_bias = bn.bias.data.clone()

    apply_kaiming_init(nn.Sequential(bn))

    # BatchNorm weights should be unchanged (not a Conv/Linear)
    assert torch.equal(bn.weight.data, original_weight)
    assert torch.equal(bn.bias.data, original_bias)


def test_weight_bounds_match_kaiming_uniform_formula() -> None:
    """Kaiming uniform bound = √(6 / ((1 + a²) * fan_in)) with a=0.01.

    ≈ √(6 / fan_in) since (1 + 0.0001) ≈ 1.
    """
    torch.manual_seed(42)
    fan_in = 400
    layer = nn.Linear(fan_in, 100)
    apply_kaiming_init(nn.Sequential(layer))

    a = 0.01  # negative_slope for leaky_relu
    gain = math.sqrt(2.0 / (1.0 + a * a))
    bound = gain * math.sqrt(3.0 / fan_in)

    # All weights should be within [-bound, bound]
    assert torch.all(layer.weight.data <= bound + 1e-6)
    assert torch.all(layer.weight.data >= -bound - 1e-6)


def test_init_is_deterministic_with_seed() -> None:
    """Same seed → same weights."""
    torch.manual_seed(42)
    layer1 = nn.Linear(50, 30)
    apply_kaiming_init(nn.Sequential(layer1))

    torch.manual_seed(42)
    layer2 = nn.Linear(50, 30)
    apply_kaiming_init(nn.Sequential(layer2))

    assert torch.equal(layer1.weight.data, layer2.weight.data)
    assert torch.equal(layer1.bias.data, layer2.bias.data)
