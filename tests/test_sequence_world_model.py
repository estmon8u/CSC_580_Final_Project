import numpy as np
import torch

from tiny_dreamer_highway.training import (
    compute_sequence_world_model_losses,
    stack_sequence_batch,
    train_sequence_world_model_step,
)
from tiny_dreamer_highway.models import TinyWorldModel
from tiny_dreamer_highway.types import Transition


def make_sequence_transition(seed: int) -> Transition:
    observation = np.full((1, 64, 64), seed, dtype=np.uint8)
    next_observation = np.full((1, 64, 64), seed + 1, dtype=np.uint8)
    action = np.asarray([seed / 10.0, seed / 20.0], dtype=np.float32)
    return Transition(
        observation=observation,
        action=action,
        reward=float(seed) / 10.0,
        next_observation=next_observation,
        done=False,
    )


def test_stack_sequence_batch_returns_expected_shapes() -> None:
    sequences = [
        [make_sequence_transition(seed) for seed in range(4)],
        [make_sequence_transition(seed + 10) for seed in range(4)],
    ]

    batch = stack_sequence_batch(sequences)

    assert batch.observations.shape == (2, 4, 1, 64, 64)
    assert batch.actions.shape == (2, 4, 2)
    assert batch.rewards.shape == (2, 4)
    assert batch.next_observations.shape == (2, 4, 1, 64, 64)
    assert batch.dones.shape == (2, 4)


def test_compute_sequence_world_model_losses_returns_outputs_per_step() -> None:
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    observations = torch.randint(0, 256, (2, 3, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 3, 2)
    rewards = torch.randn(2, 3)

    outputs, losses = compute_sequence_world_model_losses(model, observations, actions, rewards)

    assert len(outputs) == 3
    assert outputs[-1].reconstruction.shape == (2, 1, 64, 64)
    assert set(losses.keys()) == {"reconstruction_loss", "reward_loss", "kl_loss", "kl_loss_raw", "total_loss"}
    assert losses["total_loss"].ndim == 0
    assert losses["kl_loss"].item() >= 0.0


def test_train_sequence_world_model_step_updates_parameters() -> None:
    torch.manual_seed(7)
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randint(0, 256, (2, 4, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(2, 4, 2)
    rewards = torch.randn(2, 4)

    before = next(model.parameters()).detach().clone()
    _, metrics = train_sequence_world_model_step(model, optimizer, observations, actions, rewards)
    after = next(model.parameters()).detach().clone()

    assert metrics.keys() == {"reconstruction_loss", "reward_loss", "kl_loss", "kl_loss_raw", "total_loss"}
    assert metrics["total_loss"] >= 0.0
    assert metrics["kl_loss"] >= 0.0
    assert not torch.equal(before, after)