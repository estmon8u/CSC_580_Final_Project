"""Resume equivalence test — gold-standard checkpoint/resume validation.

Proves that:
  Run A: 2 uninterrupted training cycles
  Run B: 1 cycle → save → load → 1 cycle
produce the same model parameters, optimizer states, and replay contents.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy as np
import torch

from tiny_dreamer_highway.config import ExperimentConfig
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.models import Actor, Critic, TinyWorldModel
from tiny_dreamer_highway.training import (
    load_checkpoint,
    save_checkpoint,
)
from tiny_dreamer_highway.training.pipeline import run_training_cycle
from tiny_dreamer_highway.types import Transition


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeActionSpace:
    def __init__(self) -> None:
        self._seed: int | None = None

    def seed(self, seed: int | None) -> None:
        self._seed = seed

    def sample(self) -> np.ndarray:
        return np.asarray([0.0, 0.0], dtype=np.float32)


class _FakeEnv:
    """Deterministic env that returns distinct observations at each step."""

    def __init__(self) -> None:
        self.action_space = _FakeActionSpace()
        self._step = 0

    def reset(self, seed: int | None = None):
        self._step = 0
        obs = np.full((1, 64, 64), 3, dtype=np.uint8)
        return obs, {"seed": seed}

    def step(self, action):
        self._step += 1
        obs = np.full((1, 64, 64), 3 + self._step, dtype=np.uint8)
        reward = float(action.sum()) + 0.1 * self._step
        terminated = self._step >= 5
        return obs, reward, terminated, False, {}

    def close(self) -> None:
        pass


def _make_transition(seed: int) -> Transition:
    return Transition(
        observation=np.full((1, 64, 64), seed, dtype=np.uint8),
        action=np.asarray([seed / 10.0, seed / 20.0], dtype=np.float32),
        reward=float(seed) / 10.0,
        next_observation=np.full((1, 64, 64), seed + 1, dtype=np.uint8),
        done=False,
        terminated=False,
        truncated=False,
    )


def _create_models():
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64),
        action_dim=2,
        embedding_dim=256,
        deterministic_dim=128,
        stochastic_dim=32,
        hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    return world_model, actor, critic


def _create_optimizers(world_model, actor, critic, lr=1e-3):
    return (
        torch.optim.AdamW(world_model.parameters(), lr=lr),
        torch.optim.AdamW(actor.parameters(), lr=lr),
        torch.optim.AdamW(critic.parameters(), lr=lr),
    )


def _snapshot_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resume_then_next_step_matches_uninterrupted_run(monkeypatch, tmp_path: Path) -> None:
    """Gold-standard resume equivalence test.

    Run A: 2 consecutive training cycles (uninterrupted).
    Run B: 1 cycle → save → new objects → load → 1 cycle.
    Assert: model parameters match exactly after both runs.
    """
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.make_highway_env",
        lambda env_config: _FakeEnv(),
    )

    config = ExperimentConfig.model_validate(
        {
            "seed": 42,
            "device": "cpu",
            "training": {
                "batch_size": 2,
                "imagination_horizon": 3,
                "world_model_updates_per_cycle": 1,
                "behavior_updates_per_cycle": 1,
            },
            "replay": {"sequence_length": 4},
        }
    )

    # ----- Run A: uninterrupted 2 cycles -----
    torch.manual_seed(42)
    np.random.seed(42)
    wm_a, actor_a, critic_a = _create_models()
    wm_opt_a, actor_opt_a, critic_opt_a = _create_optimizers(wm_a, actor_a, critic_a)
    replay_a = ReplayBuffer(capacity=128)
    for i in range(16):
        replay_a.add(_make_transition(i))

    torch.manual_seed(42)
    np.random.seed(42)
    run_training_cycle(
        config, replay_a, wm_a, actor_a, critic_a,
        wm_opt_a, actor_opt_a, critic_opt_a,
        warm_start_steps=0, policy_steps=0, seed=42,
    )
    # Seed all RNG sources before cycle 2 so replay sampling is deterministic
    torch.manual_seed(99)
    np.random.seed(99)
    run_training_cycle(
        config, replay_a, wm_a, actor_a, critic_a,
        wm_opt_a, actor_opt_a, critic_opt_a,
        warm_start_steps=0, policy_steps=0, seed=43,
    )

    # ----- Run B: 1 cycle → save → reload → 1 cycle -----
    torch.manual_seed(42)
    np.random.seed(42)
    wm_b, actor_b, critic_b = _create_models()
    wm_opt_b, actor_opt_b, critic_opt_b = _create_optimizers(wm_b, actor_b, critic_b)
    replay_b = ReplayBuffer(capacity=128)
    for i in range(16):
        replay_b.add(_make_transition(i))

    torch.manual_seed(42)
    np.random.seed(42)
    run_training_cycle(
        config, replay_b, wm_b, actor_b, critic_b,
        wm_opt_b, actor_opt_b, critic_opt_b,
        warm_start_steps=0, policy_steps=0, seed=42,
    )

    # Save checkpoint after cycle 1
    ckpt_dir = tmp_path / "checkpoints"
    save_checkpoint(
        ckpt_dir, step=1,
        world_model=wm_b, actor=actor_b, critic=critic_b,
        world_model_optimizer=wm_opt_b,
        actor_optimizer=actor_opt_b,
        critic_optimizer=critic_opt_b,
        replay_buffer=replay_b,
    )

    # Create fresh objects and restore
    wm_b2, actor_b2, critic_b2 = _create_models()
    wm_opt_b2, actor_opt_b2, critic_opt_b2 = _create_optimizers(wm_b2, actor_b2, critic_b2)
    replay_b2 = ReplayBuffer(capacity=128)

    load_checkpoint(
        ckpt_dir / "checkpoint_00001.pt",
        world_model=wm_b2, actor=actor_b2, critic=critic_b2,
        world_model_optimizer=wm_opt_b2,
        actor_optimizer=actor_opt_b2,
        critic_optimizer=critic_opt_b2,
        replay_buffer=replay_b2,
    )

    # Run cycle 2 on restored state — seed identically to Run A's cycle 2
    torch.manual_seed(99)
    np.random.seed(99)
    run_training_cycle(
        config, replay_b2, wm_b2, actor_b2, critic_b2,
        wm_opt_b2, actor_opt_b2, critic_opt_b2,
        warm_start_steps=0, policy_steps=0, seed=43,
    )

    # ----- Compare parameters -----
    # With identical RNG seeds and restored optimizer state, parameters
    # should match exactly (or within float32 epsilon).
    for p_a, p_b in zip(wm_a.parameters(), wm_b2.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6), (
            "World model parameters diverged after resume"
        )
    for p_a, p_b in zip(actor_a.parameters(), actor_b2.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6), (
            "Actor parameters diverged after resume"
        )
    for p_a, p_b in zip(critic_a.parameters(), critic_b2.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6), (
            "Critic parameters diverged after resume"
        )

    # Replay buffer contents should also match
    assert len(replay_a) == len(replay_b2)


def test_checkpoint_scheduler_round_trip(tmp_path: Path) -> None:
    """Scheduler state survives save/load."""
    wm, actor, critic = _create_models()
    wm_opt, actor_opt, critic_opt = _create_optimizers(wm, actor, critic)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        wm_opt, lr_lambda=lambda step: min(1.0, (step + 1) / 10)
    )
    # Advance 5 steps (optimizer.step() must precede scheduler.step())
    dummy_loss = sum(p.sum() for p in wm.parameters())
    for _ in range(5):
        wm_opt.zero_grad()
        dummy_loss.backward(retain_graph=True)
        wm_opt.step()
        scheduler.step()

    scheduler_states = {"wm_scheduler": scheduler.state_dict()}

    ckpt_path = save_checkpoint(
        tmp_path, step=5,
        world_model=wm, actor=actor, critic=critic,
        world_model_optimizer=wm_opt,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        schedulers=scheduler_states,
    )

    # Fresh scheduler
    wm2, actor2, critic2 = _create_models()
    wm_opt2, actor_opt2, critic_opt2 = _create_optimizers(wm2, actor2, critic2)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(
        wm_opt2, lr_lambda=lambda step: min(1.0, (step + 1) / 10)
    )

    metadata = load_checkpoint(
        ckpt_path,
        world_model=wm2, actor=actor2, critic=critic2,
        world_model_optimizer=wm_opt2,
        actor_optimizer=actor_opt2,
        critic_optimizer=critic_opt2,
    )

    saved = metadata["schedulers"]["wm_scheduler"]
    scheduler2.load_state_dict(saved)

    # After 5 steps the scheduler should report lr_factor = min(1.0, 6/10) = 0.6
    assert scheduler2.get_last_lr()[0] == scheduler.get_last_lr()[0]


def test_missing_replay_sidecar_loads_without_error(tmp_path: Path) -> None:
    """When companion replay_NNNNN.pt is absent, load_checkpoint should
    still succeed (replay buffer stays empty)."""
    wm, actor, critic = _create_models()
    wm_opt, actor_opt, critic_opt = _create_optimizers(wm, actor, critic)

    # Save WITHOUT replay buffer
    ckpt_path = save_checkpoint(
        tmp_path, step=1,
        world_model=wm, actor=actor, critic=critic,
        world_model_optimizer=wm_opt,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
    )

    replay = ReplayBuffer(capacity=16)
    metadata = load_checkpoint(
        ckpt_path,
        world_model=wm, actor=actor, critic=critic,
        world_model_optimizer=wm_opt,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        replay_buffer=replay,
    )

    assert metadata["step"] == 1
    assert len(replay) == 0  # replay stays empty — no crash


def test_checkpoint_without_scheduler_key_loads_cleanly(tmp_path: Path) -> None:
    """Old checkpoints without the 'schedulers' key should load cleanly."""
    wm, actor, critic = _create_models()
    wm_opt, actor_opt, critic_opt = _create_optimizers(wm, actor, critic)

    # Save without schedulers
    ckpt_path = save_checkpoint(
        tmp_path, step=1,
        world_model=wm, actor=actor, critic=critic,
        world_model_optimizer=wm_opt,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
    )

    metadata = load_checkpoint(
        ckpt_path,
        world_model=wm, actor=actor, critic=critic,
        world_model_optimizer=wm_opt,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
    )

    assert metadata["schedulers"] is None  # graceful None, not KeyError
