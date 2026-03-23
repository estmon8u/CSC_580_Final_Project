"""Tests for discrete action support.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from pathlib import Path

import torch

from tiny_dreamer_highway.config import ActionConfig, EnvConfig, ExperimentConfig, load_experiment_config
from tiny_dreamer_highway.envs.highway_factory import build_highway_env_kwargs
from tiny_dreamer_highway.models import Critic, DiscreteActor, TinyWorldModel
from tiny_dreamer_highway.training.behavior_learning import imagine_trajectory, train_behavior_step


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_action_config_discrete_flag() -> None:
    continuous_cfg = ActionConfig(type="continuous")
    assert not continuous_cfg.is_discrete

    discrete_cfg = ActionConfig(type="discrete", num_actions=5)
    assert discrete_cfg.is_discrete
    assert discrete_cfg.num_actions == 5


def test_load_discrete_experiment_yaml() -> None:
    config_path = Path(__file__).resolve().parents[1] / "notebooks" / "configs" / "discrete_experiment.yaml"
    config = load_experiment_config(config_path)
    assert config.env.action.type == "discrete"
    assert config.env.action.is_discrete
    assert config.env.action.num_actions == 5


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def test_build_highway_env_kwargs_discrete_action_block() -> None:
    config = EnvConfig(action=ActionConfig(type="discrete", num_actions=5))
    kwargs = build_highway_env_kwargs(config)
    assert kwargs["action"]["type"] == "DiscreteMetaAction"
    # Should NOT have longitudinal/lateral keys:
    assert "longitudinal" not in kwargs["action"]
    assert "lateral" not in kwargs["action"]


def test_build_highway_env_kwargs_continuous_action_block() -> None:
    config = EnvConfig(action=ActionConfig(type="continuous"))
    kwargs = build_highway_env_kwargs(config)
    assert kwargs["action"]["type"] == "ContinuousAction"
    assert "longitudinal" in kwargs["action"]


# ---------------------------------------------------------------------------
# DiscreteActor model
# ---------------------------------------------------------------------------


def test_discrete_actor_output_shapes() -> None:
    actor = DiscreteActor(latent_dim=160, num_actions=5, hidden_dim=64, num_layers=1)
    features = torch.randn(4, 160)

    # Training mode → one-hot from Gumbel-Softmax
    actor.train()
    actions = actor(features)
    assert actions.shape == (4, 5)
    # Each row should sum to ~1.0 (hard one-hot via straight-through)
    assert torch.allclose(actions.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Eval mode → deterministic one-hot from argmax
    actor.eval()
    actions_eval = actor(features)
    assert actions_eval.shape == (4, 5)
    assert torch.allclose(actions_eval.sum(dim=-1), torch.ones(4), atol=1e-5)
    # Exactly one 1.0 per row
    assert (actions_eval.max(dim=-1).values == 1.0).all()


# ---------------------------------------------------------------------------
# Imagination with DiscreteActor
# ---------------------------------------------------------------------------


def test_imagine_trajectory_with_discrete_actor() -> None:
    action_dim = 5
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64),
        action_dim=action_dim,
        embedding_dim=256,
        deterministic_dim=128,
        num_categoricals=4, num_classes=8,
        hidden_dim=128,
    )
    actor = DiscreteActor(latent_dim=160, num_actions=action_dim, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    start_state = world_model.rssm.initial_state(batch_size=3)

    trajectory = imagine_trajectory(world_model, actor, critic, start_state, horizon=5)

    assert len(trajectory.states) == 5
    assert trajectory.features.shape == (5, 3, 160)
    assert trajectory.actions.shape == (5, 3, action_dim)
    assert trajectory.rewards.shape == (5, 3, 1)
    assert trajectory.values.shape == (5, 3, 1)


def test_imagine_trajectory_discrete_skips_stabilization() -> None:
    """DiscreteActor actions should pass through unmodified (no stabilization)."""
    action_dim = 5
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64),
        action_dim=action_dim,
        embedding_dim=256,
        deterministic_dim=128,
        num_categoricals=4, num_classes=8,
        hidden_dim=128,
    )
    actor = DiscreteActor(latent_dim=160, num_actions=action_dim, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    start_state = world_model.rssm.initial_state(batch_size=2)

    # Even with extreme stabilization params, discrete should be unaffected
    trajectory = imagine_trajectory(
        world_model,
        actor,
        critic,
        start_state,
        horizon=3,
        longitudinal_scale=0.01,
        lateral_scale=0.01,
        smoothing_factor=0.99,
    )

    # Actions should still be valid one-hot vectors
    for step in range(trajectory.actions.shape[0]):
        sums = trajectory.actions[step].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Behavior step with DiscreteActor
# ---------------------------------------------------------------------------


def test_train_behavior_step_with_discrete_actor() -> None:
    torch.manual_seed(42)
    action_dim = 5
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64),
        action_dim=action_dim,
        embedding_dim=256,
        deterministic_dim=128,
        num_categoricals=4, num_classes=8,
        hidden_dim=128,
    )
    actor = DiscreteActor(latent_dim=160, num_actions=action_dim, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    start_state = world_model.rssm.initial_state(batch_size=4)

    actor_before = next(actor.parameters()).detach().clone()
    world_before = next(world_model.parameters()).detach().clone()

    metrics = train_behavior_step(
        world_model,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        start_state,
        horizon=4,
    )

    actor_after = next(actor.parameters()).detach().clone()
    world_after = next(world_model.parameters()).detach().clone()

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert not torch.equal(actor_before, actor_after), "actor should be updated"
    assert torch.equal(world_before, world_after), "world model should be frozen"
