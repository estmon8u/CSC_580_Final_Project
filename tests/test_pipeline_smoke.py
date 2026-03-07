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
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)

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
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
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


def test_run_training_cycle_repeats_updates_per_cycle(monkeypatch) -> None:
    config = ExperimentConfig.model_validate(
        {
            "training": {
                "batch_size": 4,
                "imagination_horizon": 5,
                "world_model_updates_per_cycle": 3,
                "behavior_updates_per_cycle": 2,
            }
        }
    )
    replay_buffer = ReplayBuffer(capacity=128)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    for index in range(16):
        replay_buffer.add(_make_transition(index))

    world_calls = {"count": 0}
    behavior_calls = {"count": 0}

    def fake_train_sequence_world_model_step(*args, **kwargs):
        world_calls["count"] += 1
        return [], {
            "reconstruction_loss": 1.0,
            "reward_loss": 0.5,
            "kl_loss": 3.0,
            "kl_loss_raw": 2.0,
            "total_loss": 4.5,
        }

    def fake_stack_sequence_batch(sequences):
        from tiny_dreamer_highway.training.sequence_world_model_step import stack_sequence_batch
        return stack_sequence_batch(sequences)

    def fake_seed_latent_state(*args, **kwargs):
        return world_model.rssm.initial_state(batch_size=4)

    def fake_train_behavior_step(*args, **kwargs):
        behavior_calls["count"] += 1
        return {
            "actor_loss": -0.1,
            "critic_loss": 0.2,
            "imagined_reward_mean": 0.3,
            "imagined_value_mean": 0.4,
        }

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.train_sequence_world_model_step",
        fake_train_sequence_world_model_step,
    )
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.stack_sequence_batch",
        fake_stack_sequence_batch,
    )
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.seed_latent_state",
        fake_seed_latent_state,
    )
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.train_behavior_step",
        fake_train_behavior_step,
    )
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.collect_actor_transitions",
        lambda *args, **kwargs: 0,
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
        warm_start_steps=0,
        policy_steps=0,
        seed=7,
    )

    assert world_calls["count"] == 3
    assert behavior_calls["count"] == 2
    assert metrics.world_model_metrics["total_loss"] == 4.5
    assert metrics.behavior_metrics["critic_loss"] == 0.2