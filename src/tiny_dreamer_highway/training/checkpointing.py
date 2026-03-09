"""Checkpoint save/load helpers for Tiny Dreamer Highway.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn, optim

if TYPE_CHECKING:
    from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer


def checkpoint_path(checkpoint_dir: str | Path, step: int) -> Path:
    if step < 0:
        raise ValueError("step must be non-negative")
    return Path(checkpoint_dir) / f"checkpoint_{step:05d}.pt"


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    world_model: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    world_model_optimizer: optim.Optimizer,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    metrics: dict[str, Any] | None = None,
    replay_buffer: ReplayBuffer | None = None,
    schedulers: dict[str, Any] | None = None,
) -> Path:
    path = checkpoint_path(checkpoint_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "world_model": world_model.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "world_model_optimizer": world_model_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "metrics": metrics or {},
    }
    if schedulers is not None:
        payload["schedulers"] = schedulers
    torch.save(payload, path)

    # Save replay buffer to a separate file so the main checkpoint stays
    # lightweight and can be loaded even without the buffer.
    if replay_buffer is not None:
        replay_path = path.with_name(f"replay_{step:05d}.pt")
        torch.save(replay_buffer.state_dict(), replay_path)

    return path


def load_checkpoint(
    checkpoint_file: str | Path,
    world_model: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    world_model_optimizer: optim.Optimizer,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    map_location: str | torch.device = "cpu",
    replay_buffer: ReplayBuffer | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_file), map_location=map_location, weights_only=False)
    world_model.load_state_dict(checkpoint["world_model"])
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    world_model_optimizer.load_state_dict(checkpoint["world_model_optimizer"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    # Restore replay buffer from its companion file (if present)
    if replay_buffer is not None:
        step = int(checkpoint["step"])
        replay_path = Path(checkpoint_file).with_name(f"replay_{step:05d}.pt")
        if replay_path.exists():
            replay_state = torch.load(replay_path, weights_only=False)
            replay_buffer.load_state_dict(replay_state)

    return {
        "step": int(checkpoint["step"]),
        "metrics": dict(checkpoint.get("metrics", {})),
        "schedulers": checkpoint.get("schedulers"),
    }


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return None

    candidates = sorted(directory.glob("checkpoint_*.pt"))
    if not candidates:
        return None
    return candidates[-1]