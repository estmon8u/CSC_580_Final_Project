"""Replay buffer utilities for Tiny Dreamer Highway.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tiny_dreamer_highway.types import ReplayBatch, Transition


@dataclass(slots=True)
class ReplayBuffer:
    capacity: int
    transitions: list[Transition] = field(default_factory=list)
    _position: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

    def __len__(self) -> int:
        return len(self.transitions)

    def _ordered_transitions(self) -> list[Transition]:
        if len(self.transitions) < self.capacity:
            return list(self.transitions)
        return self.transitions[self._position :] + self.transitions[: self._position]

    def add(self, transition: Transition) -> None:
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def can_sample(self, batch_size: int, sequence_length: int = 1) -> bool:
        return len(self.transitions) >= max(batch_size, sequence_length)

    def sample_batch(self, batch_size: int) -> ReplayBatch:
        if not self.can_sample(batch_size=batch_size):
            raise ValueError(
                f"not enough transitions to sample a batch "
                f"(requested {batch_size}, have {len(self.transitions)})"
            )

        allow_replacement = batch_size > len(self.transitions)
        indices = np.random.choice(len(self.transitions), size=batch_size, replace=allow_replacement)
        selected = [self.transitions[index] for index in indices]
        return ReplayBatch(
            observations=np.stack([item.observation for item in selected]).astype(np.uint8),
            actions=np.stack([item.action for item in selected]).astype(np.float32),
            rewards=np.asarray([item.reward for item in selected], dtype=np.float32),
            next_observations=np.stack([item.next_observation for item in selected]).astype(np.uint8),
            dones=np.asarray([item.done for item in selected], dtype=np.bool_),
        )

    def sample_sequences(self, batch_size: int, sequence_length: int) -> list[list[Transition]]:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.can_sample(batch_size=batch_size, sequence_length=sequence_length):
            raise ValueError("not enough transitions to sample sequences")

        ordered = self._ordered_transitions()
        max_start = len(ordered) - sequence_length
        if max_start < 0:
            raise ValueError("sequence_length exceeds replay size")

        valid_start_indices = [
            start_index
            for start_index in range(max_start + 1)
            if not any(transition.done for transition in ordered[start_index : start_index + sequence_length - 1])
        ]
        if not valid_start_indices:
            raise ValueError("no valid sequences available without crossing episode boundaries")

        allow_replacement = batch_size > len(valid_start_indices)
        start_indices = np.random.choice(
            np.asarray(valid_start_indices, dtype=np.int64),
            size=batch_size,
            replace=allow_replacement,
        )
        return [
            ordered[start_index : start_index + sequence_length]
            for start_index in start_indices.tolist()
        ]
