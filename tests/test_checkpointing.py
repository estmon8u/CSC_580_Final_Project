from pathlib import Path

import torch

from tiny_dreamer_highway.models import Actor, Critic, TinyWorldModel
from tiny_dreamer_highway.training import (
    checkpoint_path,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_path_formats_step_with_padding(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, 12)
    assert path == tmp_path / "checkpoint_00012.pt"


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    world_model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    actor = Actor(latent_dim=160, action_dim=2)
    critic = Critic(latent_dim=160)
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    checkpoint_file = save_checkpoint(
        tmp_path,
        step=3,
        world_model=world_model,
        actor=actor,
        critic=critic,
        world_model_optimizer=world_optimizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        metrics={"loss": 1.23},
    )

    original_world = next(world_model.parameters()).detach().clone()
    original_actor = next(actor.parameters()).detach().clone()
    original_critic = next(critic.parameters()).detach().clone()

    with torch.no_grad():
        next(world_model.parameters()).add_(1.0)
        next(actor.parameters()).add_(1.0)
        next(critic.parameters()).add_(1.0)

    metadata = load_checkpoint(
        checkpoint_file,
        world_model=world_model,
        actor=actor,
        critic=critic,
        world_model_optimizer=world_optimizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    assert metadata == {"step": 3, "metrics": {"loss": 1.23}}
    assert torch.allclose(next(world_model.parameters()), original_world)
    assert torch.allclose(next(actor.parameters()), original_actor)
    assert torch.allclose(next(critic.parameters()), original_critic)


def test_find_latest_checkpoint_returns_highest_step(tmp_path: Path) -> None:
    first = tmp_path / "checkpoint_00001.pt"
    latest = tmp_path / "checkpoint_00010.pt"
    first.write_bytes(b"a")
    latest.write_bytes(b"b")

    assert find_latest_checkpoint(tmp_path) == latest
    assert find_latest_checkpoint(tmp_path / "missing") is None