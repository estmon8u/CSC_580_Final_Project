"""Shared typed containers for Tiny Dreamer Highway.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Transition:
    observation: NDArray[np.uint8]
    action: NDArray[np.float32]
    reward: float
    next_observation: NDArray[np.uint8]
    done: bool
    terminated: bool = False
    truncated: bool = False


@dataclass(slots=True)
class ReplayBatch:
    observations: NDArray[np.uint8]
    actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    next_observations: NDArray[np.uint8]
    dones: NDArray[np.bool_]
    terminals: NDArray[np.bool_]
    truncations: NDArray[np.bool_]


@dataclass(slots=True)
class ReplaySequenceBatch:
    observations: NDArray[np.uint8]
    actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    next_observations: NDArray[np.uint8]
    dones: NDArray[np.bool_]
    terminals: NDArray[np.bool_]
    truncations: NDArray[np.bool_]
