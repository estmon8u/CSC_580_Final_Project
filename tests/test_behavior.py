import torch

from tiny_dreamer_highway.models import Actor, Critic, TinyWorldModel
from tiny_dreamer_highway.training import imagine_trajectory, td_lambda_returns, train_behavior_step


def test_actor_and_critic_return_expected_shapes() -> None:
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    features = torch.randn(4, 160)

    actions = actor(features)
    values = critic(features)

    assert actions.shape == (4, 2)
    assert values.shape == (4, 1)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)

    value_dist = critic.distribution(features)
    log_prob = value_dist.log_prob(torch.randn(4, 1))
    assert log_prob.shape == (4,)


def test_imagine_trajectory_returns_expected_shapes() -> None:
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    start_state = world_model.rssm.initial_state(batch_size=3)

    trajectory = imagine_trajectory(world_model, actor, critic, start_state, horizon=5)

    assert len(trajectory.states) == 5
    assert trajectory.features.shape == (5, 3, 160)
    assert trajectory.actions.shape == (5, 3, 2)
    assert trajectory.rewards.shape == (5, 3, 1)
    assert trajectory.values.shape == (5, 3, 1)
    assert trajectory.continues is not None
    assert trajectory.continues.shape == (5, 3, 1)
    assert trajectory.bootstrap.shape == (3, 1)


def test_td_lambda_returns_matches_one_step_case_when_lambda_zero() -> None:
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[10.0]], [[20.0]]])

    returns = td_lambda_returns(rewards, values, discount=0.5, lambda_=0.0)

    # With λ=0, returns[t] = rewards[t] + discount * next_values[t]
    # next_values = [values[1], bootstrap=values[-1]] = [[[20.0]], [[20.0]]]
    # step 0: 1.0 + 0.5 * 20.0 = 11.0
    # step 1: 2.0 + 0.5 * 20.0 = 12.0
    expected = torch.tensor([[[11.0]], [[12.0]]])
    assert torch.allclose(returns, expected)


def test_td_lambda_returns_uses_per_step_discounts_when_provided() -> None:
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[10.0]], [[20.0]]])
    discounts = torch.tensor([[[0.5]], [[0.0]]])

    returns = td_lambda_returns(rewards, values, discount=0.99, lambda_=0.0, discounts=discounts)

    expected = torch.tensor([[[11.0]], [[2.0]]])
    assert torch.allclose(returns, expected)


def test_train_behavior_step_updates_actor_and_critic_without_changing_world_model() -> None:
    torch.manual_seed(7)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    start_state = world_model.rssm.initial_state(batch_size=4)

    world_before = next(world_model.parameters()).detach().clone()
    actor_before = next(actor.parameters()).detach().clone()
    critic_before = next(critic.parameters()).detach().clone()

    metrics = train_behavior_step(
        world_model,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        start_state,
        horizon=4,
    )

    world_after = next(world_model.parameters()).detach().clone()
    actor_after = next(actor.parameters()).detach().clone()
    critic_after = next(critic.parameters()).detach().clone()

    assert metrics.keys() == {
        "actor_loss",
        "critic_loss",
        "imagined_reward_mean",
        "imagined_value_mean",
    }
    assert not torch.equal(actor_before, actor_after)
    assert not torch.equal(critic_before, critic_after)
    assert torch.equal(world_before, world_after)
    assert all(parameter.grad is None for parameter in world_model.parameters())