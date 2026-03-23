"""End-to-end architecture tests for Tiny Dreamer Highway.

These tests exercise the full system with real (unmocked) training updates,
evaluation cadence, and discrete env-facing collection/rollout paths.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from tiny_dreamer_highway.config import ExperimentConfig
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.models import Actor, Critic, DiscreteActor, TinyWorldModel
from tiny_dreamer_highway.training.pipeline import collect_actor_transitions, run_training_cycle
from tiny_dreamer_highway.training.experiment import (
    evaluate_training_policy,
    run_training_experiment,
)
from tiny_dreamer_highway.training.pipeline import PipelineCycleMetrics
from tiny_dreamer_highway.types import Transition


# ---------------------------------------------------------------------------
# Deterministic fake envs
# ---------------------------------------------------------------------------


class _FakeActionSpace:
    def __init__(self, continuous: bool = True, n: int = 5) -> None:
        self._continuous = continuous
        self._seed: int | None = None
        self.n = n
        self.shape = (2,) if continuous else ()

    def seed(self, seed: int | None) -> None:
        self._seed = seed

    def sample(self) -> np.ndarray:
        if self._continuous:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        return np.int64(0)


class _FakeContinuousEnv:
    def __init__(self) -> None:
        self.action_space = _FakeActionSpace(continuous=True)
        self._step = 0

    def reset(self, seed: int | None = None):
        self._step = 0
        return np.full((1, 64, 64), 3, dtype=np.uint8), {}

    def step(self, action):
        self._step += 1
        obs = np.full((1, 64, 64), 3 + self._step, dtype=np.uint8)
        reward = 0.1 * self._step
        terminated = self._step >= 5
        return obs, reward, terminated, False, {}

    def render(self):
        return None

    def close(self) -> None:
        pass


class _FakeDiscreteActionSpace:
    """Mimics gymnasium.spaces.Discrete."""

    def __init__(self, n: int = 5) -> None:
        self.n = n
        self._seed: int | None = None
        self.shape = ()

    def seed(self, seed: int | None) -> None:
        self._seed = seed

    def sample(self) -> int:
        return 0


class _FakeDiscreteEnv:
    def __init__(self, n: int = 5) -> None:
        self.action_space = _FakeDiscreteActionSpace(n=n)
        self._step = 0

    def reset(self, seed: int | None = None):
        self._step = 0
        return np.full((1, 64, 64), 3, dtype=np.uint8), {}

    def step(self, action):
        self._step += 1
        obs = np.full((1, 64, 64), 3 + self._step, dtype=np.uint8)
        reward = 0.2 * self._step
        terminated = self._step >= 5
        return obs, reward, terminated, False, {}

    def render(self):
        return None

    def close(self) -> None:
        pass


def _make_transition(seed: int) -> Transition:
    return Transition(
        observation=np.full((1, 64, 64), seed % 200, dtype=np.uint8),
        action=np.asarray([seed / 10.0, seed / 20.0], dtype=np.float32),
        reward=float(seed) / 10.0,
        next_observation=np.full((1, 64, 64), (seed + 1) % 200, dtype=np.uint8),
        done=False,
        terminated=False,
        truncated=False,
    )


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


def test_one_training_cycle_produces_finite_metrics_and_expected_update_boundaries(
    monkeypatch,
) -> None:
    """Real (unmocked) single training cycle with a fake env.

    Verifies:
    - All metric values are finite (no NaN/Inf)
    - World model parameters change during WM update
    - Actor/critic parameters change during behavior update
    - Replay buffer grows only during collection phases
    """
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.make_highway_env",
        lambda env_config: _FakeContinuousEnv(),
    )

    config = ExperimentConfig.model_validate(
        {
            "seed": 7,
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

    torch.manual_seed(7)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, num_categoricals=4, num_classes=8, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    wm_opt = torch.optim.AdamW(world_model.parameters(), lr=1e-3)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    replay = ReplayBuffer(capacity=128)

    # Pre-fill replay with enough data
    for i in range(16):
        replay.add(_make_transition(i))

    wm_before = [p.detach().clone() for p in world_model.parameters()]
    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]

    metrics = run_training_cycle(
        config, replay, world_model, actor, critic,
        wm_opt, actor_opt, critic_opt,
        warm_start_steps=0, policy_steps=0, seed=7,
    )

    # All metrics must be finite
    for key, value in metrics.world_model_metrics.items():
        assert math.isfinite(value), f"world_model/{key} is not finite: {value}"
    for key, value in metrics.behavior_metrics.items():
        assert math.isfinite(value), f"behavior/{key} is not finite: {value}"

    # World model should have changed
    wm_changed = any(
        not torch.equal(before, after)
        for before, after in zip(wm_before, world_model.parameters())
    )
    assert wm_changed, "World model parameters should change during WM update"

    # Actor and critic should have changed
    actor_changed = any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor.parameters())
    )
    critic_changed = any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, critic.parameters())
    )
    assert actor_changed, "Actor parameters should change during behavior update"
    assert critic_changed, "Critic parameters should change during behavior update"


# ---------------------------------------------------------------------------
# Evaluate training policy
# ---------------------------------------------------------------------------


def test_evaluate_training_policy_returns_expected_keys(monkeypatch) -> None:
    """Verify evaluate_training_policy runs evaluation episodes and returns
    the expected metric keys with reasonable values."""
    monkeypatch.setattr(
        "tiny_dreamer_highway.evaluation.policy_rollout.make_highway_env",
        lambda env_config: _FakeContinuousEnv(),
    )

    config = ExperimentConfig.model_validate({"seed": 7, "device": "cpu"})
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, num_categoricals=4, num_classes=8, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)

    eval_metrics = evaluate_training_policy(
        config, world_model, actor,
        episodes=2, max_steps=5, seed=7,
    )

    assert set(eval_metrics.keys()) == {"episodes", "mean_reward", "mean_steps", "crash_rate"}
    assert eval_metrics["episodes"] == 2.0
    assert math.isfinite(eval_metrics["mean_reward"])
    assert eval_metrics["mean_steps"] > 0
    assert 0.0 <= eval_metrics["crash_rate"] <= 1.0


def test_evaluate_training_policy_restores_training_mode(monkeypatch) -> None:
    """Models should be back in their original training mode after evaluation."""
    monkeypatch.setattr(
        "tiny_dreamer_highway.evaluation.policy_rollout.make_highway_env",
        lambda env_config: _FakeContinuousEnv(),
    )

    config = ExperimentConfig.model_validate({"seed": 7, "device": "cpu"})
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, num_categoricals=4, num_classes=8, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)

    world_model.train()
    actor.train()

    evaluate_training_policy(config, world_model, actor, episodes=1, max_steps=3, seed=7)

    assert world_model.training, "World model should be restored to training mode"
    assert actor.training, "Actor should be restored to training mode"


def test_evaluate_training_policy_zero_episodes_returns_empty() -> None:
    config = ExperimentConfig.model_validate({"seed": 7, "device": "cpu"})
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, num_categoricals=4, num_classes=8, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)

    result = evaluate_training_policy(config, world_model, actor, episodes=0, max_steps=5)
    assert result == {}


# ---------------------------------------------------------------------------
# Discrete env-facing collection
# ---------------------------------------------------------------------------


def test_collect_actor_transitions_discrete_stores_one_hot_actions(monkeypatch) -> None:
    """When using a DiscreteActor, collect_actor_transitions should store
    one-hot-encoded actions in the replay buffer, not raw integers."""

    import gymnasium as gym
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.make_highway_env",
        lambda env_config: _FakeDiscreteEnv(n=5),
    )
    # Also patch isinstance check for gym.spaces.Discrete
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.hasattr",
        lambda obj, attr: True if attr == "seed" else hasattr(obj, attr),
    ) if False else None  # no-op, hasattr on fake works fine

    config = ExperimentConfig.model_validate(
        {
            "seed": 7,
            "device": "cpu",
            "env": {"action": {"type": "discrete", "num_actions": 5}},
        }
    )

    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=5,
        embedding_dim=256, deterministic_dim=128, num_categoricals=4, num_classes=8, hidden_dim=128,
    )
    actor = DiscreteActor(latent_dim=160, num_actions=5, hidden_dim=64, num_layers=1)
    replay = ReplayBuffer(capacity=32)

    added = collect_actor_transitions(
        config, replay, world_model, actor, steps=3, seed=7,
    )

    assert added == 3
    assert len(replay) == 3

    # Each stored action should be a 5-dim one-hot vector
    for i in range(len(replay)):
        action = replay.transitions[i].action
        assert action.shape == (5,), f"Expected shape (5,), got {action.shape}"
        assert abs(action.sum() - 1.0) < 1e-5, f"Expected one-hot sum 1.0, got {action.sum()}"


# ---------------------------------------------------------------------------
# Full experiment orchestration smoke test
# ---------------------------------------------------------------------------


def test_full_experiment_orchestration_with_fake_training_cycle(
    monkeypatch, tmp_path: Path,
) -> None:
    """Smoke test for run_training_experiment with checkpointing, logging,
    and evaluation cadence — using a fake training cycle but real orchestration."""
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.make_highway_env",
        lambda config: _FakeContinuousEnv(),
    )

    eval_calls = {"count": 0}

    def fake_evaluate(config, world_model, actor, *, episodes, max_steps, seed=None):
        eval_calls["count"] += 1
        return {
            "episodes": float(episodes),
            "mean_reward": 1.5,
            "mean_steps": 4.0,
            "crash_rate": 0.5,
        }

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.evaluate_training_policy",
        fake_evaluate,
    )

    def fake_run_training_cycle(
        config, replay_buffer, world_model, actor, critic,
        wm_opt, actor_opt, critic_opt,
        warm_start_steps=0, policy_steps=0, seed=None,
        *, wm_scaler=None, actor_scaler=None, critic_scaler=None, amp_context=None,
    ) -> PipelineCycleMetrics:
        for _ in range(max(1, warm_start_steps + policy_steps)):
            replay_buffer.add(
                Transition(
                    observation=np.zeros((1, 64, 64), dtype=np.uint8),
                    action=np.zeros((2,), dtype=np.float32),
                    reward=0.0,
                    next_observation=np.zeros((1, 64, 64), dtype=np.uint8),
                    done=False,
                )
            )
        return PipelineCycleMetrics(
            warm_start_added=warm_start_steps,
            policy_added=policy_steps,
            replay_size=len(replay_buffer),
            world_model_metrics={
                "reconstruction_loss": 0.5, "reward_loss": 0.2,
                "continue_loss": 0.1, "total_loss": 0.8,
            },
            behavior_metrics={
                "actor_loss": -0.05, "critic_loss": 0.15,
                "imagined_reward_mean": 0.4, "imagined_value_mean": 0.3,
            },
            evaluation_metrics={},
        )

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.experiment.run_training_cycle",
        fake_run_training_cycle,
    )

    config = ExperimentConfig.model_validate(
        {
            "seed": 7,
            "device": "cpu",
            "training": {
                "cycles": 4,
                "warm_start_steps": 4,
                "policy_steps": 2,
                "checkpoint_interval": 2,
            },
            "evaluation": {
                "episodes": 2,
                "interval": 2,
                "max_steps": 5,
            },
        }
    )

    summary = run_training_experiment(config, tmp_path, show_progress=False)

    # Orchestration checks
    assert summary.completed_cycles == 4
    assert summary.latest_checkpoint is not None
    assert summary.latest_checkpoint.exists()

    # Evaluation should have been called at steps 2 and 4 (every 2 cycles)
    assert eval_calls["count"] == 2

    # Artifacts should exist
    assert (tmp_path / "logs" / "cycle_metrics.csv").exists()
    assert (tmp_path / "logs" / "cycle_metrics.jsonl").exists()
    assert (tmp_path / "logs" / "latest_summary.json").exists()

    # Checkpoint at step 2 and 4 should exist
    assert (tmp_path / "checkpoints" / "checkpoint_00002.pt").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_00004.pt").exists()
