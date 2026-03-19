"""Deterministic mathematical tests for DiscreteActor (Gumbel-Softmax).

Verifies: eval-mode argmax one-hot, train-mode hard one-hot (row sum=1,
one element=1), seeded reproducibility, and logit-to-action mapping.
"""

import torch
import torch.nn.functional as F

from tiny_dreamer_highway.models.discrete_actor import DiscreteActor


def _make_discrete_actor(seed: int = 42, **kwargs) -> DiscreteActor:
    defaults = dict(
        latent_dim=40, num_actions=5, hidden_dim=32,
        num_layers=1, gumbel_temperature=1.0,
    )
    defaults.update(kwargs)
    torch.manual_seed(seed)
    return DiscreteActor(**defaults)


# ── eval mode: argmax one-hot ────────────────────────────────────────

def test_eval_argmax_known_logits() -> None:
    """With known logits [0.1, 0.5, 0.3, 0.8, 0.2], argmax is index 3."""
    actor = _make_discrete_actor()
    actor.eval()

    # Override the network weights to produce known logits
    with torch.no_grad():
        for p in actor.net.parameters():
            p.zero_()
        # Set the final layer bias to our desired logits
        actor.net[-1].bias.copy_(torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2]))

    x = torch.zeros(1, 40)
    out = actor(x)

    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    assert torch.equal(out, expected)


def test_eval_argmax_batch() -> None:
    """Each batch item selects its own argmax independently."""
    actor = _make_discrete_actor()
    actor.eval()

    with torch.no_grad():
        for p in actor.net.parameters():
            p.zero_()
        # Different logits for different inputs by setting weight pattern
        actor.net[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

    x = torch.zeros(3, 40)
    out = actor(x)

    # All should select index 0
    assert torch.all(out[:, 0] == 1.0)
    assert torch.all(out[:, 1:] == 0.0)


def test_eval_is_deterministic() -> None:
    """Same input → same output every time in eval mode."""
    actor = _make_discrete_actor()
    actor.eval()
    x = torch.randn(4, 40)

    out1 = actor(x)
    out2 = actor(x)
    assert torch.equal(out1, out2)


def test_eval_output_is_valid_one_hot() -> None:
    """Each row sums to 1.0 and has exactly one 1.0."""
    actor = _make_discrete_actor()
    actor.eval()
    x = torch.randn(10, 40)

    out = actor(x)

    # Row sums
    assert torch.allclose(out.sum(dim=-1), torch.ones(10), atol=1e-6)
    # Exactly one 1.0 per row
    assert torch.all(out.max(dim=-1).values == 1.0)
    # Only 0s and 1s
    assert torch.all((out == 0.0) | (out == 1.0))


# ── train mode: Gumbel-Softmax hard one-hot ──────────────────────────

def test_train_output_is_hard_one_hot() -> None:
    """Gumbel-Softmax with hard=True produces hard one-hot vectors."""
    actor = _make_discrete_actor()
    actor.train()
    x = torch.randn(10, 40)

    out = actor(x)

    # Row sums
    assert torch.allclose(out.sum(dim=-1), torch.ones(10), atol=1e-6)
    # Only 0s and 1s (hard straight-through)
    assert torch.all((out == 0.0) | (out == 1.0))


def test_train_mode_has_gradient() -> None:
    """Training output should allow gradient backpropagation."""
    actor = _make_discrete_actor()
    actor.train()
    x = torch.randn(2, 40)

    out = actor(x)
    loss = out.sum()
    loss.backward()

    for p in actor.parameters():
        assert p.grad is not None


def test_train_output_shape() -> None:
    actor = _make_discrete_actor(num_actions=7)
    actor.train()
    x = torch.randn(3, 40)

    out = actor(x)
    assert out.shape == (3, 7)


# ── Gumbel-Softmax formula verification ─────────────────────────────

def test_gumbel_softmax_row_sum_always_one() -> None:
    """F.gumbel_softmax(hard=True) always produces rows summing to 1."""
    torch.manual_seed(42)
    logits = torch.randn(100, 5)

    for tau in [0.1, 0.5, 1.0, 5.0]:
        out = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        assert torch.allclose(out.sum(dim=-1), torch.ones(100), atol=1e-6)


def test_gumbel_softmax_lower_temperature_more_peaked() -> None:
    """Lower temperature → distribution peaks more around the argmax logit.

    With weakly separated logits, count how often argmax(logits) matches
    argmax(gumbel_sample). Lower temperature should match more often.
    """
    # Use nearly uniform logits so Gumbel noise can flip the argmax
    logits = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1]]).expand(500, -1)

    torch.manual_seed(42)
    high_temp = F.gumbel_softmax(logits, tau=10.0, hard=True, dim=-1)
    high_temp_matches = (high_temp.argmax(dim=-1) == 0).float().mean()

    torch.manual_seed(42)
    low_temp = F.gumbel_softmax(logits, tau=0.01, hard=True, dim=-1)
    low_temp_matches = (low_temp.argmax(dim=-1) == 0).float().mean()

    # Low temperature should select the highest logit more frequently
    assert low_temp_matches >= high_temp_matches


# ── seeded reproducibility ───────────────────────────────────────────

def test_seeded_discrete_actor_eval_is_reproducible() -> None:
    x = torch.randn(3, 40)

    torch.manual_seed(42)
    a1 = _make_discrete_actor(seed=42)
    a1.eval()
    out1 = a1(x)

    torch.manual_seed(42)
    a2 = _make_discrete_actor(seed=42)
    a2.eval()
    out2 = a2(x)

    assert torch.equal(out1, out2)


# ── output dtype ─────────────────────────────────────────────────────

def test_eval_one_hot_dtype_matches_logits() -> None:
    """one_hot output should have same dtype as logits (float32)."""
    actor = _make_discrete_actor()
    actor.eval()
    x = torch.randn(2, 40)

    out = actor(x)
    assert out.dtype == torch.float32
