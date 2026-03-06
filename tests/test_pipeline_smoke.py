import numpy as np
import torch

from tiny_dreamer_highway.config import ExperimentConfig
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.models import Actor, Critic, TinyWorldModel
from tiny_dreamer_highway.training import collect_actor_transitions, run_training_cycle
from tiny_dreamer_highway.types import Transition


class _FakeActionSpace:
    def __init__(self) -> None:
        self._seed = None

    def seed(self, seed: int | None) -> None:
        self._seed = seed

    def sample(self) -> np.ndarray:
        return np.asarray([0.0, 0.0], dtype=np.float32)


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = _FakeActionSpace()
        self._step = 0

    def reset(self, seed: int | None = None):
        self._step = 0
        observation = np.full((1, 64, 64), 3, dtype=np.uint8)
        return observation, {"seed": seed}

    def step(self, action: np.ndarray):
        self._step += 1
        observation = np.full((1, 64, 64), 3 + self._step, dtype=np.uint8)
        reward = float(action.sum()) + 0.1 * self._step
        terminated = self._step >= 3
        truncated = False
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        return None


def _make_transition(seed: int) -> Transition:
    return Transition(
        observation=np.full((1, 64, 64), seed, dtype=np.uint8),
        action=np.asarray([seed / 10.0, seed / 20.0], dtype=np.float32),
        reward=float(seed) / 10.0,
        next_observation=np.full((1, 64, 64), seed + 1, dtype=np.uint8),
        done=False,
    )


def test_collect_actor_transitions_adds_policy_steps(monkeypatch) -> None:
    config = ExperimentConfig()
    replay_buffer = ReplayBuffer(capacity=32)
    world_model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    actor = Actor(latent_dim=160, action_dim=2)

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.make_highway_env",
        lambda env_config: _FakeEnv(),
    )

    added = collect_actor_transitions(
        config,
        replay_buffer,
        world_model,
        actor,
        steps=3,
        seed=7,
    )

    assert added == 3
    assert len(replay_buffer) == 3
    assert replay_buffer.transitions[0].action.shape == (2,)


def test_run_training_cycle_executes_warm_start_train_and_policy_collection(monkeypatch) -> None:
    torch.manual_seed(7)
    config = ExperimentConfig()
    replay_buffer = ReplayBuffer(capacity=128)
    world_model = TinyWorldModel(observation_shape=(1, 64, 64), action_dim=2)
    actor = Actor(latent_dim=160, action_dim=2)
    critic = Critic(latent_dim=160)
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    def fake_collect_random_transitions(env_config, buffer, steps: int, seed: int | None = None) -> int:
        for index in range(steps):
            buffer.add(_make_transition(index))
        return steps

    def fake_collect_actor_transitions(config, buffer, world_model, actor, steps: int, seed: int | None = None) -> int:
        for index in range(steps):
            buffer.add(_make_transition(index + 100))
        return steps

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.collect_random_transitions",
        fake_collect_random_transitions,
    )
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.collect_actor_transitions",
        fake_collect_actor_transitions,
    )

    metrics = run_training_cycle(
        config,
        replay_buffer,
        world_model,
        actor,
        critic,
        world_optimizer,
        actor_optimizer,
        critic_optimizer,
        warm_start_steps=16,
        policy_steps=3,
        seed=7,
    )

    assert metrics.warm_start_added == 16
    assert metrics.policy_added == 3
    assert metrics.replay_size == 19
    assert set(metrics.world_model_metrics.keys()) == {
        "reconstruction_loss",
        "reward_loss",
            "kl_loss",
            "kl_loss_raw",
        "total_loss",
    }
    assert set(metrics.behavior_metrics.keys()) == {
        "actor_loss",
        "critic_loss",
        "imagined_reward_mean",
        "imagined_value_mean",
    }