import numpy as np
import pytest

from tiny_dreamer_highway.config import EnvConfig
from tiny_dreamer_highway.data.collect_random_rollouts import collect_random_transitions
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer


class FakeActionSpace:
    def sample(self) -> np.ndarray:
        return np.asarray([0.25, -0.5], dtype=np.float32)


class FakeEnv:
    def __init__(self) -> None:
        self.action_space = FakeActionSpace()
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False

    def reset(self):
        self.reset_calls += 1
        observation = np.full((4, 4), self.reset_calls, dtype=np.uint8)
        return observation, {"reset_calls": self.reset_calls}

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
    added = collect_random_transitions(EnvConfig(), replay_buffer, steps=5)

    assert added == 5
    assert len(replay_buffer) == 5
    assert fake_env.reset_calls == 3
    assert fake_env.step_calls == 5
    assert fake_env.closed is True

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
    collect_random_transitions(EnvConfig(), replay_buffer, steps=4)

    done_flags = [transition.done for transition in replay_buffer.transitions]
    assert done_flags == [False, True, False, True]