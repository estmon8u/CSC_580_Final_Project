import torch

from tiny_dreamer_highway.models import TinyWorldModel
from tiny_dreamer_highway.training import compute_world_model_losses, gaussian_kl_divergence, train_world_model_step


def test_tiny_world_model_forward_returns_expected_shapes() -> None:
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
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
    assert output.predicted_continue is not None
    assert output.predicted_continue.shape == (4, 1)


def test_compute_world_model_losses_returns_expected_keys() -> None:
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    observations = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(3, 2)
    rewards = torch.randn(3)

    output = model(observations, actions)
    losses = compute_world_model_losses(output, observations, rewards, target_dones=torch.zeros(3))

    assert set(losses.keys()) == {
        "reconstruction_loss",
        "reward_loss",
        "continue_loss",
        "kl_loss",
        "kl_loss_raw",
        "total_loss",
    }
    assert losses["total_loss"].ndim == 0
    assert losses["kl_loss"].item() >= 0.0
    assert losses["continue_loss"].item() >= 0.0


def test_train_world_model_step_runs_optimizer_step() -> None:
    torch.manual_seed(7)
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randint(0, 256, (4, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(4, 2)
    rewards = torch.randn(4)
    dones = torch.zeros(4)

    before = next(model.parameters()).detach().clone()
    _, metrics = train_world_model_step(model, optimizer, observations, actions, rewards, dones=dones)
    after = next(model.parameters()).detach().clone()

    assert metrics.keys() == {
        "reconstruction_loss",
        "reward_loss",
        "continue_loss",
        "kl_loss",
        "kl_loss_raw",
        "total_loss",
    }
    assert metrics["total_loss"] >= 0.0
    assert metrics["kl_loss"] >= 0.0
    assert not torch.equal(before, after)


def test_gaussian_kl_divergence_is_zero_for_identical_distributions() -> None:
    mean = torch.randn(4, 32)
    std = torch.ones(4, 32)
    kl = gaussian_kl_divergence(mean, std, mean, std)
    assert kl.item() < 1e-5


def test_gaussian_kl_divergence_is_positive_for_different_distributions() -> None:
    posterior_mean = torch.zeros(4, 32)
    posterior_std = torch.ones(4, 32)
    prior_mean = torch.ones(4, 32)
    prior_std = torch.ones(4, 32) * 2.0
    kl = gaussian_kl_divergence(posterior_mean, posterior_std, prior_mean, prior_std)
    assert kl.item() > 0.0