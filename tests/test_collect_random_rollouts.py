import numpy as np
import pytest

from tiny_dreamer_highway.config import EnvConfig
from tiny_dreamer_highway.data.collect_random_rollouts import collect_random_transitions
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer


class FakeActionSpace:
    def __init__(self) -> None:
        self.last_seed: int | None = None

    def sample(self) -> np.ndarray:
        return np.asarray([0.25, -0.5], dtype=np.float32)

    def seed(self, seed: int) -> None:
        self.last_seed = seed


class FakeEnv:
    def __init__(self) -> None:
        self.action_space = FakeActionSpace()
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False

    def reset(self, seed: int | None = None):
        self.reset_calls += 1
        observation = np.full((4, 4), self.reset_calls, dtype=np.uint8)
        return observation, {"reset_calls": self.reset_calls, "seed": seed}

    def step(self, action: np.ndarray):
        self.step_calls += 1
        next_observation = np.full((4, 4), 10 + self.step_calls, dtype=np.uint8)
        reward = float(self.step_calls)
        terminated = self.step_calls % 2 == 0
        truncated = False
        return next_observation, reward, terminated, truncated, {"action_sum": float(action.sum())}

    def close(self) -> None:
        self.closed = True


def test_collect_random_transitions_adds_expected_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeEnv()
    monkeypatch.setattr(
        "tiny_dreamer_highway.data.collect_random_rollouts.make_highway_env",
        lambda config: fake_env,
    )

    replay_buffer = ReplayBuffer(capacity=16)
    added = collect_random_transitions(EnvConfig(), replay_buffer, steps=5, seed=7)

    assert added == 5
    assert len(replay_buffer) == 5
    assert fake_env.reset_calls == 3
    assert fake_env.step_calls == 5
    assert fake_env.closed is True
    assert fake_env.action_space.last_seed == 7

    first = replay_buffer.transitions[0]
    assert first.observation.dtype == np.uint8
    assert first.next_observation.dtype == np.uint8
    assert first.action.dtype == np.float32
    assert isinstance(first.reward, float)


def test_collect_random_transitions_resets_after_done(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = FakeEnv()
    monkeypatch.setattr(
        "tiny_dreamer_highway.data.collect_random_rollouts.make_highway_env",
        lambda config: fake_env,
    )

    replay_buffer = ReplayBuffer(capacity=16)
    collect_random_transitions(EnvConfig(), replay_buffer, steps=4, seed=7)

    done_flags = [transition.done for transition in replay_buffer.transitions]
    assert done_flags == [False, True, False, True]


class FakeVectorActionSpace:
    def sample(self) -> np.ndarray:
        return np.zeros((2, 2), dtype=np.float32)


class FakeVectorEnv:
    """Mimics SyncVectorEnv API for testing."""

    def __init__(self) -> None:
        self.num_envs = 2
        self.action_space = FakeVectorActionSpace()
        self._step_count = np.zeros(2, dtype=int)
        self.closed = False

    def reset(self, seed=None):
        self._step_count[:] = 0
        observations = np.full((2, 4, 4), 1, dtype=np.uint8)
        return observations, {}

    def step(self, actions):
        self._step_count += 1
        observations = np.full((2, 4, 4), 10, dtype=np.uint8)
        rewards = np.ones(2, dtype=np.float64)
        terminations = self._step_count >= 2
        truncations = np.zeros(2, dtype=bool)
        infos: dict = {}
        if terminations.any():
            infos["_final_observation"] = terminations.copy()
            infos["final_observation"] = [
                np.full((4, 4), 99, dtype=np.uint8) if terminations[i] else None
                for i in range(2)
            ]
            self._step_count[terminations] = 0
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self.closed = True


def test_collect_random_transitions_vectorized(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_vec_env = FakeVectorEnv()
    monkeypatch.setattr(
        "tiny_dreamer_highway.data.collect_random_rollouts.make_vectorized_highway_env",
        lambda config: fake_vec_env,
    )

    config = EnvConfig(num_envs=2)
    replay_buffer = ReplayBuffer(capacity=32)
    added = collect_random_transitions(config, replay_buffer, steps=6, seed=7)

    # ceil(6 / 2) = 3 iterations × 2 envs = 6 transitions
    assert added == 6
    assert len(replay_buffer) == 6
    assert fake_vec_env.closed is True


def test_vectorized_collection_stores_per_env_contiguously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each env's transitions must be stored in a contiguous block.

    With 2 envs and 3 iterations the buffer should look like:
        [Env0_S1, Env0_S2, Env0_S3, Env1_S1, Env1_S2, Env1_S3]
    NOT the interleaved layout:
        [Env0_S1, Env1_S1, Env0_S2, Env1_S2, Env0_S3, Env1_S3]
    """

    class TaggedVectorEnv:
        """Each env produces observations tagged with its env index."""

        def __init__(self) -> None:
            self.num_envs = 2
            self.action_space = FakeVectorActionSpace()
            self._step_count = np.zeros(2, dtype=int)
            self.closed = False

        def reset(self, seed=None):
            self._step_count[:] = 0
            # Tag observations: env 0 → pixel value 100, env 1 → 200
            obs = np.zeros((2, 4, 4), dtype=np.uint8)
            obs[0] = 100
            obs[1] = 200
            return obs, {}

        def step(self, actions):
            self._step_count += 1
            # Tag next-obs with env index + step: env 0 → 10+step, env 1 → 20+step
            obs = np.zeros((2, 4, 4), dtype=np.uint8)
            obs[0] = 10 + self._step_count[0]
            obs[1] = 20 + self._step_count[1]
            rewards = np.array([float(self._step_count[0]), float(self._step_count[1])])
            terminations = np.zeros(2, dtype=bool)
            truncations = np.zeros(2, dtype=bool)
            return obs, rewards, terminations, truncations, {}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "tiny_dreamer_highway.data.collect_random_rollouts.make_vectorized_highway_env",
        lambda config: TaggedVectorEnv(),
    )

    config = EnvConfig(num_envs=2)
    replay_buffer = ReplayBuffer(capacity=32)
    added = collect_random_transitions(config, replay_buffer, steps=6, seed=7)
    assert added == 6

    transitions = replay_buffer.transitions
    # First 3 belong to env 0, last 3 belong to env 1
    # Env 0 observations should have pixel value 100 (initial reset)
    assert transitions[0].observation.flat[0] == 100, (
        "First block should be env 0's trajectory"
    )
    assert transitions[1].observation.flat[0] == 11, (
        "Second transition should be env 0's second step"
    )
    assert transitions[2].observation.flat[0] == 12, (
        "Third transition should be env 0's third step"
    )
    # Env 1 block starts at index 3
    assert transitions[3].observation.flat[0] == 200, (
        "Fourth transition should be env 1's first step (reset obs)"
    )
    assert transitions[4].observation.flat[0] == 21, (
        "Fifth transition should be env 1's second step"
    )
    assert transitions[5].observation.flat[0] == 22, (
        "Sixth transition should be env 1's third step"
    )