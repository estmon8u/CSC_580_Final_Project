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
