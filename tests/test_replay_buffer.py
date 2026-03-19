import numpy as np

from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.types import Transition


def make_transition(seed: int) -> Transition:
    observation = np.full((4, 4), seed, dtype=np.uint8)
    next_observation = np.full((4, 4), seed + 1, dtype=np.uint8)
    action = np.asarray([seed, seed + 0.5], dtype=np.float32)
    return Transition(
        observation=observation,
        action=action,
        reward=float(seed),
        next_observation=next_observation,
        done=bool(seed % 2),
    )


def test_replay_buffer_respects_capacity() -> None:
    buffer = ReplayBuffer(capacity=3)
    for seed in range(5):
        buffer.add(make_transition(seed))

    assert len(buffer) == 3


def test_replay_buffer_sample_batch_shapes() -> None:
    buffer = ReplayBuffer(capacity=8)
    for seed in range(8):
        buffer.add(make_transition(seed))

    batch = buffer.sample_batch(batch_size=4)
    assert batch.observations.shape == (4, 4, 4)
    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4,)
    assert batch.next_observations.shape == (4, 4, 4)
    assert batch.dones.shape == (4,)


def test_replay_buffer_sample_sequences_length() -> None:
    buffer = ReplayBuffer(capacity=10)
    for seed in range(10):
        transition = make_transition(seed)
        transition.done = False
        buffer.add(transition)

    sequences = buffer.sample_sequences(batch_size=3, sequence_length=4)
    assert len(sequences) == 3
    assert all(len(sequence) == 4 for sequence in sequences)


def test_replay_buffer_sample_sequences_do_not_cross_episode_boundaries() -> None:
    buffer = ReplayBuffer(capacity=8)
    for seed in range(8):
        transition = make_transition(seed)
        transition.done = seed == 2
        buffer.add(transition)

    sequences = buffer.sample_sequences(batch_size=4, sequence_length=3)
    for sequence in sequences:
        assert not any(transition.done for transition in sequence[:-1])


def test_replay_buffer_can_sample_sequences_requires_valid_contiguous_sequence() -> None:
    buffer = ReplayBuffer(capacity=8)
    for seed in range(8):
        transition = make_transition(seed)
        transition.done = seed % 2 == 0
        buffer.add(transition)

    assert not buffer.can_sample(batch_size=4, sequence_length=3)


def test_replay_buffer_sample_sequences_use_chronological_order_after_wraparound() -> None:
    buffer = ReplayBuffer(capacity=4)
    for seed in range(6):
        transition = make_transition(seed)
        transition.done = False
        buffer.add(transition)

    sequence = buffer.sample_sequences(batch_size=1, sequence_length=3)[0]
    observations = [int(transition.observation[0, 0]) for transition in sequence]
    assert observations in ([2, 3, 4], [3, 4, 5])


def test_replay_buffer_state_dict_round_trip() -> None:
    buffer = ReplayBuffer(capacity=8)
    for seed in range(6):
        buffer.add(make_transition(seed))

    state = buffer.state_dict()
    assert state["capacity"] == 8
    assert state["size"] == 6
    assert state["position"] == 6

    restored = ReplayBuffer(capacity=8)
    restored.load_state_dict(state)
    assert len(restored) == 6
    assert restored._position == 6
    for original, loaded in zip(buffer.transitions, restored.transitions):
        assert original.reward == loaded.reward
        assert (original.observation == loaded.observation).all()
        assert (original.next_observation == loaded.next_observation).all()


def test_replay_buffer_add_batch_stores_all_transitions() -> None:
    buffer = ReplayBuffer(capacity=16)
    n = 4
    obs = np.arange(n * 2 * 2, dtype=np.uint8).reshape(n, 2, 2)
    act = np.ones((n, 2), dtype=np.float32)
    rew = np.arange(n, dtype=np.float32)
    next_obs = obs + 1
    dones = np.array([False, True, False, False])
    terminated = np.array([False, True, False, False])
    truncated = np.array([False, False, False, False])

    buffer.add_batch(obs, act, rew, next_obs, dones, terminated, truncated)

    assert len(buffer) == 4
    assert buffer.transitions[1].done is True
    assert buffer.transitions[1].terminated is True
    assert buffer.transitions[0].reward == 0.0
    assert buffer.transitions[3].reward == 3.0
    assert (buffer.transitions[0].observation == obs[0]).all()


def test_replay_buffer_add_batch_wraps_around() -> None:
    buffer = ReplayBuffer(capacity=4)
    obs = np.zeros((3, 2, 2), dtype=np.uint8)
    act = np.zeros((3, 1), dtype=np.float32)
    rew = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    next_obs = obs.copy()
    dones = np.zeros(3, dtype=bool)
    terminated = np.zeros(3, dtype=bool)
    truncated = np.zeros(3, dtype=bool)

    buffer.add_batch(obs, act, rew, next_obs, dones, terminated, truncated)
    assert len(buffer) == 3

    # Second batch wraps around
    rew2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    buffer.add_batch(obs, act, rew2, next_obs, dones, terminated, truncated)
    assert len(buffer) == 4  # capped at capacity
    # The newest entries should be accessible
    rewards = [t.reward for t in buffer.transitions]
    assert 6.0 in rewards
