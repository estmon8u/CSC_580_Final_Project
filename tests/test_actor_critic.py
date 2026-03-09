"""Dedicated Actor and Critic tests.

Verify shape contracts, train/eval mode semantics, action range invariants,
distribution validity, and numerical stability.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tiny_dreamer_highway.models import Actor, Critic, DiscreteActor


# ---------------------------------------------------------------------------
# Actor (tanh-normal continuous)
# ---------------------------------------------------------------------------


class TestActorShapeAndRange:
    """Basic output contracts for the continuous Actor."""

    LATENT_DIM = 160
    ACTION_DIM = 2
    BATCH = 8

    def _make_actor(self, **kwargs) -> Actor:
        defaults = dict(latent_dim=self.LATENT_DIM, action_dim=self.ACTION_DIM,
                        hidden_dim=64, num_layers=1)
        defaults.update(kwargs)
        return Actor(**defaults)

    def test_train_mode_output_shape(self) -> None:
        actor = self._make_actor()
        actor.train()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out = actor(x)
        assert out.shape == (self.BATCH, self.ACTION_DIM)

    def test_eval_mode_output_shape(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out = actor(x)
        assert out.shape == (self.BATCH, self.ACTION_DIM)

    def test_actions_in_range_neg1_to_1(self) -> None:
        """Both train and eval should produce actions in [-1, 1]."""
        actor = self._make_actor()
        x = torch.randn(self.BATCH, self.LATENT_DIM)

        actor.train()
        train_out = actor(x)
        assert train_out.min() >= -1.0 - 1e-5
        assert train_out.max() <= 1.0 + 1e-5

        actor.eval()
        eval_out = actor(x)
        assert eval_out.min() >= -1.0 - 1e-5
        assert eval_out.max() <= 1.0 + 1e-5

    def test_eval_is_deterministic(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out1 = actor(x)
        out2 = actor(x)
        assert torch.equal(out1, out2)

    def test_train_mode_has_gradient(self) -> None:
        actor = self._make_actor()
        actor.train()
        x = torch.randn(self.BATCH, self.LATENT_DIM, requires_grad=True)
        out = actor(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_output_changes_when_input_changes(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x1 = torch.randn(1, self.LATENT_DIM)
        x2 = x1 + 1.0  # different input
        out1 = actor(x1)
        out2 = actor(x2)
        assert not torch.equal(out1, out2)

    def test_no_nan_on_large_input(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x = torch.randn(4, self.LATENT_DIM) * 100.0
        out = actor(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class TestCriticShapeAndDistribution:
    """Basic output contracts for the Critic."""

    LATENT_DIM = 160
    BATCH = 8

    def _make_critic(self, **kwargs) -> Critic:
        defaults = dict(latent_dim=self.LATENT_DIM, hidden_dim=64, num_layers=1)
        defaults.update(kwargs)
        return Critic(**defaults)

    def test_forward_shape(self) -> None:
        critic = self._make_critic()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out = critic(x)
        assert out.shape == (self.BATCH, 1)

    def test_distribution_log_prob_shape(self) -> None:
        critic = self._make_critic()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        dist = critic.distribution(x)
        lp = dist.log_prob(torch.randn(self.BATCH, 1))
        assert lp.shape == (self.BATCH,)

    def test_distribution_rsample_finite(self) -> None:
        critic = self._make_critic()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        dist = critic.distribution(x)
        sample = dist.rsample()
        assert not torch.isnan(sample).any()
        assert not torch.isinf(sample).any()

    def test_eval_is_deterministic(self) -> None:
        critic = self._make_critic()
        critic.eval()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out1 = critic(x)
        out2 = critic(x)
        assert torch.equal(out1, out2)

    def test_no_nan_on_large_input(self) -> None:
        critic = self._make_critic()
        x = torch.randn(4, self.LATENT_DIM) * 100.0
        out = critic(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows(self) -> None:
        critic = self._make_critic()
        x = torch.randn(self.BATCH, self.LATENT_DIM, requires_grad=True)
        out = critic(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_updates_on_value_targets(self) -> None:
        """Optimiser step should change parameters in the direction
        that reduces the target MSE."""
        critic = self._make_critic()
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        targets = torch.ones(self.BATCH, 1) * 5.0

        loss_before = F.mse_loss(critic(x), targets)

        for _ in range(10):
            opt.zero_grad()
            loss = F.mse_loss(critic(x), targets)
            loss.backward()
            opt.step()

        loss_after = F.mse_loss(critic(x), targets)
        assert loss_after.item() < loss_before.item()


# ---------------------------------------------------------------------------
# DiscreteActor
# ---------------------------------------------------------------------------


class TestDiscreteActorShapeAndMode:
    """Basic contracts for DiscreteActor."""

    LATENT_DIM = 160
    NUM_ACTIONS = 5
    BATCH = 8

    def _make_actor(self, **kwargs) -> DiscreteActor:
        defaults = dict(latent_dim=self.LATENT_DIM, num_actions=self.NUM_ACTIONS,
                        hidden_dim=64, num_layers=1)
        defaults.update(kwargs)
        return DiscreteActor(**defaults)

    def test_train_output_shape(self) -> None:
        actor = self._make_actor()
        actor.train()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out = actor(x)
        assert out.shape == (self.BATCH, self.NUM_ACTIONS)

    def test_eval_output_shape(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out = actor(x)
        assert out.shape == (self.BATCH, self.NUM_ACTIONS)

    def test_eval_output_is_one_hot(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out = actor(x)
        # Each row sums to 1 and has exactly one 1.0
        assert torch.allclose(out.sum(dim=-1), torch.ones(self.BATCH))
        assert (out.max(dim=-1).values == 1.0).all()

    def test_eval_is_deterministic(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out1 = actor(x)
        out2 = actor(x)
        assert torch.equal(out1, out2)

    def test_train_mode_has_gradient(self) -> None:
        actor = self._make_actor()
        actor.train()
        x = torch.randn(self.BATCH, self.LATENT_DIM, requires_grad=True)
        out = actor(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_train_output_is_hard_one_hot(self) -> None:
        """Gumbel-Softmax hard=True should produce one-hot even in training."""
        actor = self._make_actor()
        actor.train()
        x = torch.randn(self.BATCH, self.LATENT_DIM)
        out = actor(x)
        # Should be 0 or 1 valued
        assert ((out == 0) | (out == 1)).all()
        assert torch.allclose(out.sum(dim=-1), torch.ones(self.BATCH))

    def test_no_nan_on_large_input(self) -> None:
        actor = self._make_actor()
        actor.eval()
        x = torch.randn(4, self.LATENT_DIM) * 100.0
        out = actor(x)
        assert not torch.isnan(out).any()
