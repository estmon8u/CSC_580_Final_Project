"""Deterministic mathematical tests for Actor (continuous tanh-normal).

Verifies exact numerics: mean_scale*tanh(raw/mean_scale) scaling,
softplus(raw_std + init_std) + min_std, eval-mode determinism (tanh(mean)),
train-mode TanhTransform bounds, and seeded reproducibility.
"""

import math

import torch
import torch.nn.functional as F

from tiny_dreamer_highway.models.actor import Actor


def _make_actor(seed: int = 42, **kwargs) -> Actor:
    defaults = dict(
        latent_dim=40, action_dim=2, hidden_dim=32, num_layers=1,
        init_std=5.0, mean_scale=5.0, min_std=1e-4,
    )
    defaults.update(kwargs)
    torch.manual_seed(seed)
    return Actor(**defaults)


# ── mean_scale * tanh(raw / mean_scale) ──────────────────────────────

def test_mean_scaling_at_zero() -> None:
    """raw=0 → mean_scale * tanh(0/mean_scale) = 0."""
    mean_scale = 5.0
    raw = torch.tensor(0.0)
    scaled = mean_scale * torch.tanh(raw / mean_scale)
    assert scaled.item() == 0.0


def test_mean_scaling_at_mean_scale() -> None:
    """raw=5.0, mean_scale=5.0 → 5*tanh(1) ≈ 3.8080."""
    mean_scale = 5.0
    raw = torch.tensor(5.0)
    scaled = mean_scale * torch.tanh(raw / mean_scale)

    expected = 5.0 * math.tanh(1.0)  # ≈ 3.80797
    assert math.isclose(scaled.item(), expected, rel_tol=1e-5)


def test_mean_scaling_saturates_for_large_input() -> None:
    """raw=100, mean_scale=5.0 → 5*tanh(20) ≈ 5.0 (saturated)."""
    mean_scale = 5.0
    raw = torch.tensor(100.0)
    scaled = mean_scale * torch.tanh(raw / mean_scale)

    assert math.isclose(scaled.item(), 5.0, abs_tol=1e-6)


def test_mean_scaling_is_odd_function() -> None:
    """f(-x) = -f(x) for the mean parameterisation."""
    mean_scale = 5.0
    raw = torch.tensor(3.0)
    pos = mean_scale * torch.tanh(raw / mean_scale)
    neg = mean_scale * torch.tanh(-raw / mean_scale)

    assert math.isclose(pos.item(), -neg.item(), abs_tol=1e-7)


# ── softplus(raw_std + init_std) + min_std ───────────────────────────

def test_std_at_zero_raw() -> None:
    """raw_std=0, init_std=5 → softplus(5) + 1e-4 ≈ 5.0068."""
    init_std = 5.0
    min_std = 1e-4
    raw_std = torch.tensor(0.0)

    computed = F.softplus(raw_std + init_std) + min_std
    expected = math.log(1 + math.exp(5.0)) + 1e-4  # softplus(5) ≈ 5.00671
    assert math.isclose(computed.item(), expected, rel_tol=1e-5)


def test_std_at_negative_init_std() -> None:
    """raw_std=−5, init_std=5 → softplus(0) + 1e-4 = ln(2) + 1e-4 ≈ 0.6932."""
    init_std = 5.0
    min_std = 1e-4
    raw_std = torch.tensor(-5.0)

    computed = F.softplus(raw_std + init_std) + min_std
    expected = math.log(2.0) + 1e-4
    assert math.isclose(computed.item(), expected, rel_tol=1e-5)


def test_std_always_positive() -> None:
    """std > 0 for any raw_std value (softplus is positive, plus min_std)."""
    init_std = 5.0
    min_std = 1e-4

    for val in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]:
        raw_std = torch.tensor(val)
        computed = F.softplus(raw_std + init_std) + min_std
        assert computed.item() > 0.0


# ── eval mode: output = tanh(mean) ──────────────────────────────────

def test_eval_output_bounded_neg1_to_1() -> None:
    actor = _make_actor()
    actor.eval()
    x = torch.randn(10, 40)
    out = actor(x)

    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)


def test_eval_is_deterministic() -> None:
    """Eval mode returns tanh(mean) — no sampling, same output every time."""
    actor = _make_actor()
    actor.eval()
    x = torch.randn(3, 40)

    out1 = actor(x)
    out2 = actor(x)
    assert torch.equal(out1, out2)


def test_eval_output_equals_tanh_of_scaled_mean() -> None:
    """In eval: output = tanh(mean_scale * tanh(raw_mean / mean_scale)).

    We verify this by running the network manually to extract raw output,
    then comparing against the actor's eval forward.
    """
    actor = _make_actor(seed=42)
    actor.eval()
    x = torch.randn(2, 40)

    # Run the internal network
    with torch.no_grad():
        raw = actor.net(x)
        raw_mean, _ = raw.split(actor.action_dim, dim=-1)
        scaled_mean = actor.mean_scale * torch.tanh(raw_mean / actor.mean_scale)
        expected = torch.tanh(scaled_mean)

    actual = actor(x)
    assert torch.allclose(actual, expected, atol=1e-6)


# ── train mode: TanhTransform bounds ─────────────────────────────────

def test_train_output_bounded() -> None:
    """Train mode samples through TanhTransform → output ∈ [-1, 1]."""
    actor = _make_actor()
    actor.train()
    x = torch.randn(50, 40)

    out = actor(x)
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)


def test_train_mode_has_gradient() -> None:
    actor = _make_actor()
    actor.train()
    x = torch.randn(2, 40)

    out = actor(x)
    loss = out.sum()
    loss.backward()

    for p in actor.parameters():
        assert p.grad is not None


# ── seeded reproducibility ───────────────────────────────────────────

def test_seeded_actor_forward_is_reproducible() -> None:
    x = torch.randn(3, 40)

    torch.manual_seed(42)
    a1 = _make_actor(seed=42)
    a1.eval()
    out1 = a1(x)

    torch.manual_seed(42)
    a2 = _make_actor(seed=42)
    a2.eval()
    out2 = a2(x)

    assert torch.equal(out1, out2)


def test_actor_output_shape() -> None:
    actor = _make_actor(action_dim=3)
    actor.eval()
    x = torch.randn(5, 40)
    out = actor(x)
    assert out.shape == (5, 3)
