import torch

from tiny_dreamer_highway.models import TinyWorldModel
from tiny_dreamer_highway.training import compute_world_model_losses, train_world_model_step


def test_tiny_world_model_forward_returns_expected_shapes() -> None:
    model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    observations = torch.randint(0, 256, (4, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(4, 2)

    output = model(observations, actions)

    assert output.embedding.shape == (4, 256)
    assert output.prior_state.deterministic is not None
    assert output.prior_state.stochastic is not None
    assert output.posterior_state.deterministic is not None
    assert output.posterior_state.stochastic is not None
    assert output.posterior_state.features.shape == (4, 160)
    assert output.reconstruction.shape == (4, 1, 64, 64)
    assert output.predicted_reward.shape == (4, 1)


def test_compute_world_model_losses_returns_expected_keys() -> None:
    model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    observations = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(3, 2)
    rewards = torch.randn(3)

    output = model(observations, actions)
    losses = compute_world_model_losses(output, observations, rewards)

    assert set(losses.keys()) == {"reconstruction_loss", "reward_loss", "total_loss"}
    assert losses["total_loss"].ndim == 0


def test_train_world_model_step_runs_optimizer_step() -> None:
    torch.manual_seed(7)
    model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randint(0, 256, (4, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(4, 2)
    rewards = torch.randn(4)

    before = next(model.parameters()).detach().clone()
    _, metrics = train_world_model_step(model, optimizer, observations, actions, rewards)
    after = next(model.parameters()).detach().clone()

    assert metrics.keys() == {"reconstruction_loss", "reward_loss", "total_loss"}
    assert metrics["total_loss"] >= 0.0
    assert not torch.equal(before, after)