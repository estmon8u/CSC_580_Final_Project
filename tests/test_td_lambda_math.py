"""Deterministic mathematical tests for TD-lambda returns.

Verifies exact hand-computed values for λ=0 (1-step TD), λ=1 (Monte Carlo),
λ=0.5 (mixed), λ=0.95 (Dreamer default), and per-step discounts from the
continue model.
"""

import torch

from tiny_dreamer_highway.training.behavior_learning import (
    td_lambda_returns,
    trajectory_loss_weights,
    weighted_mean,
)


# ── λ=0: 1-step TD ──────────────────────────────────────────────────

def test_lambda_zero_three_step() -> None:
    """λ=0 → returns[t] = r_t + γ * v_{t+1}.

    rewards = [1, 2, 3], values = [10, 20, 30], γ=0.9
    next_values = [20, 30, 30] (bootstrap = values[-1])
    returns[0] = 1 + 0.9*20 = 19.0
    returns[1] = 2 + 0.9*30 = 29.0
    returns[2] = 3 + 0.9*30 = 30.0
    """
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    values = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])

    returns = td_lambda_returns(rewards, values, discount=0.9, lambda_=0.0)

    expected = torch.tensor([[[19.0]], [[29.0]], [[30.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)


def test_lambda_zero_with_explicit_bootstrap() -> None:
    """λ=0 with explicit bootstrap ≠ values[-1].

    rewards = [1, 2], values = [10, 20], bootstrap = 50, γ=0.5
    next_values = [20, 50]
    returns[0] = 1 + 0.5*20 = 11.0
    returns[1] = 2 + 0.5*50 = 27.0
    """
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[10.0]], [[20.0]]])
    bootstrap = torch.tensor([[50.0]])

    returns = td_lambda_returns(rewards, values, bootstrap=bootstrap, discount=0.5, lambda_=0.0)

    expected = torch.tensor([[[11.0]], [[27.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)


# ── λ=1: Monte Carlo returns ────────────────────────────────────────

def test_lambda_one_reduces_to_monte_carlo() -> None:
    """λ=1 → returns[t] = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^H * bootstrap.

    rewards = [1, 2, 3], bootstrap = 0, γ=1.0
    returns[2] = 3 + 1.0*0 = 3.0
    returns[1] = 2 + 1.0*3.0 = 5.0
    returns[0] = 1 + 1.0*5.0 = 6.0
    """
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    values = torch.tensor([[[99.0]], [[99.0]], [[99.0]]])  # ignored when λ=1
    bootstrap = torch.tensor([[0.0]])

    returns = td_lambda_returns(rewards, values, bootstrap=bootstrap, discount=1.0, lambda_=1.0)

    expected = torch.tensor([[[6.0]], [[5.0]], [[3.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)


def test_lambda_one_with_discount() -> None:
    """λ=1, γ=0.5: returns are discounted sum.

    rewards = [4, 2], bootstrap = 10, γ=0.5
    returns[1] = 2 + 0.5 * 10 = 7.0
    returns[0] = 4 + 0.5 * 7.0 = 7.5
    """
    rewards = torch.tensor([[[4.0]], [[2.0]]])
    values = torch.tensor([[[0.0]], [[0.0]]])
    bootstrap = torch.tensor([[10.0]])

    returns = td_lambda_returns(rewards, values, bootstrap=bootstrap, discount=0.5, lambda_=1.0)

    expected = torch.tensor([[[7.5]], [[7.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)


# ── λ=0.5: mixed TD and MC ──────────────────────────────────────────

def test_lambda_half_two_step() -> None:
    """λ=0.5, γ=1.0 for clarity.

    rewards = [1, 2], values = [10, 20], bootstrap = 20, γ=1.0, λ=0.5
    next_values = [20, 20]

    Backward from step 1:
      blended_1 = (1-0.5)*next_values[1] + 0.5*next_return
      next_return starts as bootstrap = 20
      blended_1 = 0.5*20 + 0.5*20 = 20.0
      returns[1] = 2 + 1.0*20 = 22.0
      next_return = 22.0

    Step 0:
      blended_0 = 0.5*next_values[0] + 0.5*next_return
      = 0.5*20 + 0.5*22 = 21.0
      returns[0] = 1 + 1.0*21 = 22.0
    """
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[10.0]], [[20.0]]])

    returns = td_lambda_returns(rewards, values, discount=1.0, lambda_=0.5)

    expected = torch.tensor([[[22.0]], [[22.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)


# ── λ=0.95 (Dreamer default) ────────────────────────────────────────

def test_lambda_095_two_step() -> None:
    """λ=0.95, γ=0.99.

    rewards = [1, 2], values = [10, 20], bootstrap = 20
    next_values = [20, 20]

    Step 1:
      blended = 0.05*20 + 0.95*20 = 20.0
      returns[1] = 2 + 0.99*20 = 21.80

    Step 0:
      blended = 0.05*20 + 0.95*21.80 = 1.0 + 20.71 = 21.71
      returns[0] = 1 + 0.99*21.71 = 22.4929
    """
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[10.0]], [[20.0]]])

    returns = td_lambda_returns(rewards, values, discount=0.99, lambda_=0.95)

    # Step 1: 2 + 0.99 * ((0.05*20) + (0.95*20)) = 2 + 0.99*20 = 21.8
    step1 = 2.0 + 0.99 * ((1 - 0.95) * 20.0 + 0.95 * 20.0)
    # Step 0: 1 + 0.99 * ((0.05*20) + (0.95*step1))
    step0 = 1.0 + 0.99 * ((1 - 0.95) * 20.0 + 0.95 * step1)
    expected = torch.tensor([[[step0]], [[step1]]])
    assert torch.allclose(returns, expected, atol=1e-4)


# ── Per-step discounts (continue model) ──────────────────────────────

def test_per_step_discounts_zero_discount_stops_propagation() -> None:
    """discount=0 at step t → returns[t] = r_t (no future contribution)."""
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    values = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    discounts = torch.tensor([[[0.9]], [[0.0]], [[0.9]]])

    returns = td_lambda_returns(rewards, values, discount=0.99, lambda_=0.0, discounts=discounts)

    # Step 2: r + d*v_next = 3 + 0.9*30 = 30.0
    # Step 1: r + d*v_next = 2 + 0.0*30 = 2.0  (discount=0 kills it)
    # Step 0: r + d*v_next = 1 + 0.9*20 = 19.0
    expected = torch.tensor([[[19.0]], [[2.0]], [[30.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)


def test_per_step_discounts_mixed() -> None:
    """Per-step discounts with λ=0 and varying discount values."""
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[10.0]], [[20.0]]])
    discounts = torch.tensor([[[0.5]], [[0.8]]])

    returns = td_lambda_returns(rewards, values, discount=0.99, lambda_=0.0, discounts=discounts)

    # Step 1: r + d*bootstrap = 2 + 0.8*20 = 18.0
    # Step 0: r + d*next_value = 1 + 0.5*20 = 11.0
    expected = torch.tensor([[[11.0]], [[18.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)


# ── trajectory_loss_weights ──────────────────────────────────────────

def test_weights_uniform_discount() -> None:
    """weights[t] = γ^t with uniform discount."""
    rewards = torch.zeros(4, 1, 1)

    weights = trajectory_loss_weights(rewards, discount=0.5)

    expected = torch.tensor([[[1.0]], [[0.5]], [[0.25]], [[0.125]]])
    assert torch.allclose(weights, expected, atol=1e-6)


def test_weights_per_step_discounts() -> None:
    """weights[t] = ∏_{i=0..t-1} d_i (cumulative product).

    d = [0.5, 0.25, 0.1]
    weights = [1.0, 0.5, 0.5*0.25=0.125]
    """
    rewards = torch.zeros(3, 1, 1)
    discounts = torch.tensor([[[0.5]], [[0.25]], [[0.1]]])

    weights = trajectory_loss_weights(rewards, discounts=discounts)

    expected = torch.tensor([[[1.0]], [[0.5]], [[0.125]]])
    assert torch.allclose(weights, expected, atol=1e-6)


def test_weights_discount_one_gives_uniform() -> None:
    """γ=1 → all weights are 1.0."""
    rewards = torch.zeros(5, 1, 1)

    weights = trajectory_loss_weights(rewards, discount=1.0)

    expected = torch.ones(5, 1, 1)
    assert torch.allclose(weights, expected, atol=1e-6)


# ── weighted_mean ────────────────────────────────────────────────────

def test_weighted_mean_uniform_weights() -> None:
    """Uniform weights → regular mean."""
    values = torch.tensor([2.0, 4.0, 6.0])
    weights = torch.ones(3)

    result = weighted_mean(values, weights)
    assert torch.allclose(result, torch.tensor(4.0), atol=1e-6)


def test_weighted_mean_non_uniform() -> None:
    """weighted_mean = Σ(v*w) / Σ(w).

    values = [10, 20], weights = [1, 3]
    result = (10*1 + 20*3) / (1+3) = 70/4 = 17.5
    """
    values = torch.tensor([10.0, 20.0])
    weights = torch.tensor([1.0, 3.0])

    result = weighted_mean(values, weights)
    assert torch.allclose(result, torch.tensor(17.5), atol=1e-6)


def test_weighted_mean_with_zero_weight() -> None:
    """Zero-weighted items are excluded from the mean."""
    values = torch.tensor([100.0, 5.0])
    weights = torch.tensor([0.0, 1.0])

    result = weighted_mean(values, weights)
    assert torch.allclose(result, torch.tensor(5.0), atol=1e-6)


# ── batch dimension ─────────────────────────────────────────────────

def test_td_lambda_multiple_batch_items() -> None:
    """Verifies each batch item is computed independently.

    B=2, T=2, λ=0, γ=1.0
    Batch 0: rewards=[1,2], values=[10,20] → returns=[1+20, 2+20] = [21, 22]
    Batch 1: rewards=[3,4], values=[30,40] → returns=[3+40, 4+40] = [43, 44]
    """
    rewards = torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]])  # (T=2, B=2, 1)
    values = torch.tensor([[[10.0], [30.0]], [[20.0], [40.0]]])

    returns = td_lambda_returns(rewards, values, discount=1.0, lambda_=0.0)

    expected = torch.tensor([[[21.0], [43.0]], [[22.0], [44.0]]])
    assert torch.allclose(returns, expected, atol=1e-5)
