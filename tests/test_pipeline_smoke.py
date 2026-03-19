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
        terminated=False,
        truncated=False,
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


class _FakeVectorEnv:
    """Mimics the gymnasium.vector.SyncVectorEnv API for testing."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self._step_count = np.zeros(num_envs, dtype=int)

    def reset(self, seed=None):
        self._step_count[:] = 0
        observations = np.full((self.num_envs, 1, 64, 64), 3, dtype=np.uint8)
        return observations, {}

    def step(self, actions):
        self._step_count += 1
        observations = np.full((self.num_envs, 1, 64, 64), 5, dtype=np.uint8)
        rewards = np.ones(self.num_envs, dtype=np.float64) * 0.1
        # Terminate envs where step_count >= 3
        terminations = self._step_count >= 3
        truncations = np.zeros(self.num_envs, dtype=bool)

        infos: dict = {}
        if terminations.any():
            infos["_final_observation"] = terminations.copy()
            final_obs = [None] * self.num_envs
            for i in range(self.num_envs):
                if terminations[i]:
                    final_obs[i] = np.full((1, 64, 64), 99, dtype=np.uint8)
            infos["final_observation"] = final_obs
            # Auto-reset: set step counter to 0 for done envs
            self._step_count[terminations] = 0

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        pass

    @property
    def action_space(self):
        return _FakeActionSpace()


def test_collect_actor_transitions_vectorized(monkeypatch) -> None:
    config = ExperimentConfig.model_validate({"env": {"num_envs": 2}})
    replay_buffer = ReplayBuffer(capacity=64)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.make_vectorized_highway_env",
        lambda env_config: _FakeVectorEnv(num_envs=2),
    )

    added = collect_actor_transitions(
        config, replay_buffer, world_model, actor, steps=6, seed=7,
    )

    # ceil(6 / 2) = 3 iterations × 2 envs = 6 transitions
    assert added == 6
    assert len(replay_buffer) == 6
    assert replay_buffer.transitions[0].action.shape == (2,)


def test_collect_actor_transitions_vectorized_stores_per_env_contiguously(monkeypatch) -> None:
    """Verify that vectorized actor collection writes each env's
    trajectory as a contiguous block, not interleaved across envs."""

    class TaggedFakeVectorEnv:
        """Each env produces pixel values tagged with env index."""

        def __init__(self, num_envs: int) -> None:
            self.num_envs = num_envs
            self._step_count = np.zeros(num_envs, dtype=int)

        def reset(self, seed=None):
            self._step_count[:] = 0
            obs = np.zeros((self.num_envs, 1, 64, 64), dtype=np.uint8)
            obs[0] = 100  # env 0 reset tag
            obs[1] = 200  # env 1 reset tag
            return obs, {}

        def step(self, actions):
            self._step_count += 1
            obs = np.zeros((self.num_envs, 1, 64, 64), dtype=np.uint8)
            obs[0] = 10 + self._step_count[0]
            obs[1] = 20 + self._step_count[1]
            rewards = np.ones(self.num_envs, dtype=np.float64) * 0.1
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            return obs, rewards, terminations, truncations, {}

        def close(self) -> None:
            pass

        @property
        def action_space(self):
            return _FakeActionSpace()

    config = ExperimentConfig.model_validate({"env": {"num_envs": 2}})
    replay_buffer = ReplayBuffer(capacity=64)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.make_vectorized_highway_env",
        lambda env_config: TaggedFakeVectorEnv(num_envs=2),
    )

    added = collect_actor_transitions(
        config, replay_buffer, world_model, actor, steps=6, seed=7,
    )
    assert added == 6
    transitions = replay_buffer.transitions

    # First 3 transitions must all be from env 0 (contiguous)
    assert transitions[0].observation.flat[0] == 100, "env 0 reset obs"
    assert transitions[1].observation.flat[0] == 11, "env 0 step 1 obs"
    assert transitions[2].observation.flat[0] == 12, "env 0 step 2 obs"
    # Last 3 transitions must all be from env 1 (contiguous)
    assert transitions[3].observation.flat[0] == 200, "env 1 reset obs"
    assert transitions[4].observation.flat[0] == 21, "env 1 step 1 obs"
    assert transitions[5].observation.flat[0] == 22, "env 1 step 2 obs"


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
        "reconstruction_mse",
        "observation_log_prob",
        "reward_loss",
	    "continue_loss",
	    "kl_loss",
	    "kl_loss_raw",
        "overshooting_kl_loss",
        "overshooting_feature_mse",
        "overshooting_pairs",
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
            "reconstruction_mse": 0.1,
            "observation_log_prob": -1.0,
            "reward_loss": 0.5,
	        "continue_loss": 0.1,
            "kl_loss": 3.0,
            "kl_loss_raw": 2.0,
            "overshooting_kl_loss": 0.3,
            "overshooting_feature_mse": 0.2,
            "overshooting_pairs": 6.0,
            "total_loss": 4.5,
        }

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


def test_world_model_training_consumes_next_observations_not_observations(monkeypatch) -> None:
    """Prove that the pipeline passes *next_observations* (post-action) to the
    world-model sequence training step, not the pre-action observations.

    This is the key replay-alignment invariant: the RSSM advances the
    deterministic state with `action_t` *before* conditioning on the
    observation embedding, so the observation paired with `action_t` must
    be `next_obs_t` (the observation that results from taking `action_t`).

    We stamp `observations` and `next_observations` with distinguishable
    pixel values so the test can tell which array was actually consumed.
    """
    config = ExperimentConfig.model_validate(
        {
            "training": {
                "batch_size": 2,
                "imagination_horizon": 5,
                "world_model_updates_per_cycle": 1,
                "behavior_updates_per_cycle": 1,
            },
            "replay": {"sequence_length": 4},
        }
    )
    replay_buffer = ReplayBuffer(capacity=128)
    OBS_VALUE = 10   # distinguishable pre-action pixel value
    NEXT_VALUE = 200 # distinguishable post-action pixel value

    for idx in range(16):
        replay_buffer.add(
            Transition(
                observation=np.full((1, 64, 64), OBS_VALUE, dtype=np.uint8),
                action=np.asarray([idx / 10.0, idx / 20.0], dtype=np.float32),
                reward=float(idx) / 10.0,
                next_observation=np.full((1, 64, 64), NEXT_VALUE, dtype=np.uint8),
                done=False,
                terminated=False,
                truncated=False,
            )
        )

    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    captured = {}

    def spy_train_sequence_world_model_step(
        model, optimizer, observations, actions, rewards, **kwargs
    ):
        captured["observations"] = observations.clone()
        captured["actions"] = actions.clone()
        captured["rewards"] = rewards.clone()
        # Return a plausible no-op result so the pipeline proceeds
        return [], {
            "reconstruction_loss": 1.0,
            "reconstruction_mse": 0.1,
            "observation_log_prob": -1.0,
            "reward_loss": 0.5,
            "continue_loss": 0.1,
            "kl_loss": 3.0,
            "kl_loss_raw": 2.0,
            "overshooting_kl_loss": 0.3,
            "overshooting_feature_mse": 0.2,
            "overshooting_pairs": 6.0,
            "total_loss": 4.5,
        }

    def fake_seed_latent_state(*args, **kwargs):
        return world_model.rssm.initial_state(batch_size=2)

    def fake_train_behavior_step(*args, **kwargs):
        return {
            "actor_loss": -0.1,
            "critic_loss": 0.2,
            "imagined_reward_mean": 0.3,
            "imagined_value_mean": 0.4,
        }

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.train_sequence_world_model_step",
        spy_train_sequence_world_model_step,
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

    run_training_cycle(
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

    # The critical assertion: every pixel in `captured["observations"]`
    # must be NEXT_VALUE (200), not OBS_VALUE (10).
    obs_tensor = captured["observations"]
    assert obs_tensor.shape[0] == 2  # batch
    assert obs_tensor.shape[1] == 4  # sequence_length
    # Convert to uint8 for pixel comparison (observations are loaded as uint8)
    unique_values = obs_tensor.unique().tolist()
    assert NEXT_VALUE in unique_values, (
        f"Expected next_observations pixel value {NEXT_VALUE} in world-model "
        f"training input, but got unique values {unique_values}"
    )
    assert OBS_VALUE not in unique_values, (
        f"Pipeline incorrectly passed pre-action observations (pixel value "
        f"{OBS_VALUE}) to world-model training instead of next_observations"
    )


def test_run_training_cycle_auto_tops_up_random_data_until_sequences_exist(monkeypatch) -> None:
    config = ExperimentConfig.model_validate(
        {
            "env": {"max_episode_steps": 10},
            "training": {
                "batch_size": 2,
                "imagination_horizon": 5,
                "world_model_updates_per_cycle": 1,
                "behavior_updates_per_cycle": 1,
            },
            "replay": {"sequence_length": 4},
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

    collect_calls = {"count": 0}

    def fake_collect_random_transitions(env_config, buffer, steps: int, seed: int | None = None) -> int:
        collect_calls["count"] += 1
        if collect_calls["count"] == 1:
            for index in range(steps):
                transition = _make_transition(index)
                transition = Transition(
                    observation=transition.observation,
                    action=transition.action,
                    reward=transition.reward,
                    next_observation=transition.next_observation,
                    done=(index % 2 == 1),
                    terminated=False,
                    truncated=(index % 2 == 1),
                )
                buffer.add(transition)
            return steps

        for index in range(steps):
            buffer.add(_make_transition(index + 100))
        return steps

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.collect_random_transitions",
        fake_collect_random_transitions,
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
        warm_start_steps=4,
        policy_steps=0,
        seed=7,
    )

    assert collect_calls["count"] >= 2
    assert metrics.warm_start_added > 4
    assert metrics.replay_size >= metrics.warm_start_added


def test_run_training_cycle_rejects_impossible_sequence_length(monkeypatch) -> None:
    config = ExperimentConfig.model_validate(
        {
            "env": {"max_episode_steps": 10},
            "training": {"batch_size": 2},
            "replay": {"sequence_length": 12},
        }
    )
    replay_buffer = ReplayBuffer(capacity=32)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.collect_random_transitions",
        lambda *args, **kwargs: 0,
    )

    try:
        run_training_cycle(
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
    except ValueError as exc:
        assert "sequence_length=12 exceeds max_episode_steps=10" in str(exc)
    else:
        raise AssertionError("expected ValueError for impossible sequence length")