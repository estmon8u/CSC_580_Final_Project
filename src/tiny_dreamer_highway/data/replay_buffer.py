"""Ring-buffer replay storage for off-policy sequence sampling.

The ``ReplayBuffer`` stores transitions in preallocated NumPy arrays
for cache-efficient access and fast vectorized batch sampling.  Key
design points:

* **Lazy allocation** — arrays are allocated on the first ``add()``
  call so no observation/action shape is required at construction.
* **Ring-buffer** — when the buffer reaches ``capacity``, new
  transitions overwrite the oldest ones.
* **Sequence sampling** — ``sample_sequence_batch()`` returns
  contiguous windows of length ``sequence_length`` that do not span
  episode boundaries (validated via a cached validity bitmap).
* **Serialization** — ``state_dict()`` / ``load_state_dict()`` enable
  checkpoint persistence of the replay buffer.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot

Performance notes
-----------------
Internal storage uses preallocated NumPy arrays rather than a
``list[Transition]``.  Arrays are lazily allocated on the first
``add()`` call so no observation/action shape is required at
construction.  Sequence-start validity is cached and invalidated
on ``add()``.  A fast ``sample_sequence_batch()`` method returns
a ready-to-use ``ReplaySequenceBatch`` via vectorised fancy
indexing — no Python loops in the sampling hot-path.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tiny_dreamer_highway.types import ReplayBatch, ReplaySequenceBatch, Transition


class ReplayBuffer:
    """Ring-buffer backed by preallocated NumPy arrays.

    Arrays are lazily allocated on the first ``add()`` call, so no
    observation / action shape needs to be known at construction.
    """

    __slots__ = (
        "capacity",
        "_size",
        "_position",
        "_obs",
        "_act",
        "_rew",
        "_next_obs",
        "_dones",
        "_terminated",
        "_truncated",
        "_valid_starts_cache",
    )

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._size: int = 0
        self._position: int = 0
        self._obs: NDArray | None = None
        self._act: NDArray[np.float32] | None = None
        self._rew: NDArray[np.float32] | None = None
        self._next_obs: NDArray | None = None
        self._dones: NDArray[np.bool_] | None = None
        self._terminated: NDArray[np.bool_] | None = None
        self._truncated: NDArray[np.bool_] | None = None
        self._valid_starts_cache: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Internal allocation and indexing
    # ------------------------------------------------------------------

    def _allocate(
        self,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
        obs_dtype: np.dtype = np.dtype(np.uint8),
    ) -> None:
        """Pre-allocate storage once the shapes are known."""
        cap = self.capacity
        self._obs = np.empty((cap, *obs_shape), dtype=obs_dtype)
        self._act = np.empty((cap, *act_shape), dtype=np.float32)
        self._rew = np.empty(cap, dtype=np.float32)
        self._next_obs = np.empty((cap, *obs_shape), dtype=obs_dtype)
        self._dones = np.empty(cap, dtype=np.bool_)
        self._terminated = np.empty(cap, dtype=np.bool_)
        self._truncated = np.empty(cap, dtype=np.bool_)

    def _logical_to_physical(
        self,
        logical: NDArray[np.int64] | int,
    ) -> NDArray[np.int64] | int:
        """Map chronological (logical) indices to ring-buffer positions."""
        if self._size < self.capacity:
            return logical
        return (logical + self._position) % self.capacity

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        if self._obs is None:
            self._allocate(
                transition.observation.shape,
                transition.action.shape,
                obs_dtype=transition.observation.dtype,
            )

        idx = self._position
        self._obs[idx] = transition.observation
        self._act[idx] = transition.action
        self._rew[idx] = transition.reward
        self._next_obs[idx] = transition.next_observation
        self._dones[idx] = transition.done
        self._terminated[idx] = transition.terminated
        self._truncated[idx] = transition.truncated

        self._position = (idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

        # Invalidate cached sequence-start indices
        self._valid_starts_cache.clear()

    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        """Add *N* transitions in one vectorised write."""
        batch_size = observations.shape[0]
        if batch_size == 0:
            return
        if batch_size > self.capacity:
            raise ValueError("Cannot add a batch larger than the replay buffer capacity")

        if self._obs is None:
            self._allocate(
                observations.shape[1:],
                actions.shape[1:],
                obs_dtype=observations.dtype,
            )

        end_idx = self._position + batch_size
        if end_idx <= self.capacity:
            # Fits without wrapping — contiguous memcpy
            self._obs[self._position : end_idx] = observations
            self._act[self._position : end_idx] = actions
            self._rew[self._position : end_idx] = rewards
            self._next_obs[self._position : end_idx] = next_observations
            self._dones[self._position : end_idx] = dones
            self._terminated[self._position : end_idx] = terminated
            self._truncated[self._position : end_idx] = truncated
        else:
            # Wraps around — split into two contiguous chunks
            first_part = self.capacity - self._position
            second_part = batch_size - first_part

            self._obs[self._position :] = observations[:first_part]
            self._act[self._position :] = actions[:first_part]
            self._rew[self._position :] = rewards[:first_part]
            self._next_obs[self._position :] = next_observations[:first_part]
            self._dones[self._position :] = dones[:first_part]
            self._terminated[self._position :] = terminated[:first_part]
            self._truncated[self._position :] = truncated[:first_part]

            self._obs[:second_part] = observations[first_part:]
            self._act[:second_part] = actions[first_part:]
            self._rew[:second_part] = rewards[first_part:]
            self._next_obs[:second_part] = next_observations[first_part:]
            self._dones[:second_part] = dones[first_part:]
            self._terminated[:second_part] = terminated[first_part:]
            self._truncated[:second_part] = truncated[first_part:]

        self._position = end_idx % self.capacity
        self._size = min(self.capacity, self._size + batch_size)
        self._valid_starts_cache.clear()

    # ------------------------------------------------------------------
    # Backward-compatible property (test / debug only)
    # ------------------------------------------------------------------

    @property
    def transitions(self) -> list[Transition]:
        """Reconstruct ``list[Transition]`` in chronological order.

        After ring-buffer wrap-around the oldest entry sits at
        ``_position``, so we return elements in logical (oldest-first)
        order.

        Provided for backward compatibility with tests and serialisation.
        **Do not use on the training hot-path.**
        """
        if self._obs is None:
            return []
        if self._size < self.capacity:
            indices = range(self._size)
        else:
            indices = list(range(self._position, self.capacity)) + list(range(self._position))
        return [
            Transition(
                observation=self._obs[i],
                action=self._act[i],
                reward=float(self._rew[i]),
                next_observation=self._next_obs[i],
                done=bool(self._dones[i]),
                terminated=bool(self._terminated[i]),
                truncated=bool(self._truncated[i]),
            )
            for i in indices
        ]

    # ------------------------------------------------------------------
    # Sequence-start cache (vectorised via cumulative sum)
    # ------------------------------------------------------------------

    def valid_sequence_start_indices(self, sequence_length: int) -> list[int]:
        """Return chronological start indices for contiguous non-terminal windows.

        Results are cached and invalidated on ``add()``.
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if sequence_length in self._valid_starts_cache:
            return self._valid_starts_cache[sequence_length]

        if self._size < sequence_length:
            self._valid_starts_cache[sequence_length] = []
            return []

        if sequence_length == 1:
            result = list(range(self._size))
            self._valid_starts_cache[sequence_length] = result
            return result

        # Build chronological done mask
        if self._size < self.capacity:
            ordered_dones = self._dones[: self._size]
        else:
            ordered_dones = np.concatenate(
                [self._dones[self._position :], self._dones[: self._position]]
            )

        max_start = self._size - sequence_length
        window = sequence_length - 1

        # Cumulative-sum trick: window_sum[s] == 0 ⟹ no done in [s, s+window)
        done_int = ordered_dones.astype(np.int32)
        cs = np.empty(self._size + 1, dtype=np.int32)
        cs[0] = 0
        np.cumsum(done_int, out=cs[1:])

        starts = np.arange(max_start + 1, dtype=np.int64)
        window_sums = cs[starts + window] - cs[starts]
        valid: list[int] = starts[window_sums == 0].tolist()

        self._valid_starts_cache[sequence_length] = valid
        return valid

    # ------------------------------------------------------------------
    # Sampling predicates
    # ------------------------------------------------------------------

    def can_sample(self, batch_size: int, sequence_length: int = 1) -> bool:
        if sequence_length <= 1:
            return self._size >= max(batch_size, sequence_length)
        return len(self.valid_sequence_start_indices(sequence_length)) > 0

    # ------------------------------------------------------------------
    # Batch sampling (vectorised — no Python loop)
    # ------------------------------------------------------------------

    def sample_batch(self, batch_size: int) -> ReplayBatch:
        if not self.can_sample(batch_size=batch_size):
            raise ValueError(
                f"not enough transitions to sample a batch "
                f"(requested {batch_size}, have {self._size})"
            )

        assert self._obs is not None
        allow_replacement = batch_size > self._size
        indices = np.random.choice(self._size, size=batch_size, replace=allow_replacement)

        return ReplayBatch(
            observations=self._obs[indices],
            actions=self._act[indices],
            rewards=self._rew[indices],
            next_observations=self._next_obs[indices],
            dones=self._dones[indices],
            terminals=self._terminated[indices],
            truncations=self._truncated[indices],
        )

    # ------------------------------------------------------------------
    # Sequence sampling
    # ------------------------------------------------------------------

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> list[list[Transition]]:
        """Sample sequences as ``list[list[Transition]]`` (backward compat).

        Prefer :meth:`sample_sequence_batch` on the training hot-path.
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.can_sample(batch_size=batch_size, sequence_length=sequence_length):
            raise ValueError("not enough transitions to sample sequences")

        valid_starts = self.valid_sequence_start_indices(sequence_length)
        if not valid_starts:
            raise ValueError(
                "no valid sequences available without crossing episode boundaries"
            )

        allow_replacement = batch_size > len(valid_starts)
        chosen = np.random.choice(
            np.asarray(valid_starts, dtype=np.int64),
            size=batch_size,
            replace=allow_replacement,
        )

        assert self._obs is not None
        offsets = np.arange(sequence_length, dtype=np.int64)
        result: list[list[Transition]] = []
        for s in chosen:
            logical = int(s) + offsets
            physical = self._logical_to_physical(logical)
            result.append(
                [
                    Transition(
                        observation=self._obs[p],
                        action=self._act[p],
                        reward=float(self._rew[p]),
                        next_observation=self._next_obs[p],
                        done=bool(self._dones[p]),
                        terminated=bool(self._terminated[p]),
                        truncated=bool(self._truncated[p]),
                    )
                    for p in physical
                ]
            )
        return result

    def sample_sequence_batch(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> ReplaySequenceBatch:
        """Fast vectorised sequence sampling — no Python loops.

        Returns a ready-to-use :class:`ReplaySequenceBatch` built
        entirely via NumPy fancy indexing.
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.can_sample(batch_size=batch_size, sequence_length=sequence_length):
            raise ValueError("not enough transitions to sample sequences")

        valid_starts = self.valid_sequence_start_indices(sequence_length)
        if not valid_starts:
            raise ValueError(
                "no valid sequences available without crossing episode boundaries"
            )

        allow_replacement = batch_size > len(valid_starts)
        chosen = np.random.choice(
            np.asarray(valid_starts, dtype=np.int64),
            size=batch_size,
            replace=allow_replacement,
        )

        assert self._obs is not None
        offsets = np.arange(sequence_length, dtype=np.int64)
        logical = chosen[:, None] + offsets[None, :]  # (B, T)
        physical = self._logical_to_physical(logical)  # (B, T)

        return ReplaySequenceBatch(
            observations=self._obs[physical],
            actions=self._act[physical],
            rewards=self._rew[physical],
            next_observations=self._next_obs[physical],
            dones=self._dones[physical],
            terminals=self._terminated[physical],
            truncations=self._truncated[physical],
        )

    # ------------------------------------------------------------------
    # Serialisation helpers for checkpoint / resume
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the buffer for checkpointing."""
        if self._obs is None:
            return {
                "format": "tensorized",
                "capacity": self.capacity,
                "size": 0,
                "position": 0,
            }
        return {
            "format": "tensorized",
            "capacity": self.capacity,
            "size": self._size,
            "position": self._position,
            "observations": self._obs[: self._size].copy(),
            "actions": self._act[: self._size].copy(),
            "rewards": self._rew[: self._size].copy(),
            "next_observations": self._next_obs[: self._size].copy(),
            "dones": self._dones[: self._size].copy(),
            "terminated": self._terminated[: self._size].copy(),
            "truncated": self._truncated[: self._size].copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay buffer contents from a checkpoint snapshot.

        Handles both the new *tensorized* format and the legacy format
        that stored ``list[Transition]``.
        """
        saved_cap = state.get("capacity", self.capacity)
        if saved_cap != self.capacity:
            warnings.warn(
                f"Replay buffer capacity mismatch: checkpoint has {saved_cap}, "
                f"current config has {self.capacity}. Buffer will be loaded but may "
                "behave unexpectedly if the saved data exceeds the new capacity.",
                stacklevel=2,
            )

        if state.get("format") == "tensorized":
            size = int(state["size"])
            if size == 0:
                self._size = 0
                self._position = 0
                self._valid_starts_cache.clear()
                return

            obs = state["observations"]
            self._allocate(obs.shape[1:], state["actions"].shape[1:], obs_dtype=obs.dtype)
            n = min(size, self.capacity)
            self._obs[:n] = state["observations"][:n]
            self._act[:n] = state["actions"][:n]
            self._rew[:n] = state["rewards"][:n]
            self._next_obs[:n] = state["next_observations"][:n]
            self._dones[:n] = state["dones"][:n]
            self._terminated[:n] = state["terminated"][:n]
            self._truncated[:n] = state["truncated"][:n]
            self._size = n
            self._position = int(state["position"]) if n == size else n % self.capacity
        else:
            # Legacy format: list[Transition]
            transitions = state["transitions"]
            if not transitions:
                return

            first = transitions[0]
            self._allocate(
                first.observation.shape,
                first.action.shape,
                obs_dtype=first.observation.dtype,
            )
            n = min(len(transitions), self.capacity)
            for i in range(n):
                t = transitions[i]
                self._obs[i] = t.observation
                self._act[i] = t.action
                self._rew[i] = t.reward
                self._next_obs[i] = t.next_observation
                self._dones[i] = t.done
                self._terminated[i] = t.terminated
                self._truncated[i] = t.truncated
            self._size = n
            self._position = int(state["position"])

        self._valid_starts_cache.clear()
