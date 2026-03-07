"""Typed experiment configuration for Tiny Dreamer Highway.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ActionConfig(BaseModel):
    type: Literal["continuous"] = "continuous"
    longitudinal: bool = True
    lateral: bool = True
    longitudinal_scale: float = Field(default=1.0, gt=0.0, le=1.0)
    lateral_scale: float = Field(default=0.35, gt=0.0, le=1.0)
    smoothing_factor: float = Field(default=0.6, ge=0.0, lt=1.0)


class EnvConfig(BaseModel):
    env_id: str = "highway-v0"
    observation_height: int = Field(default=64, ge=32, le=256)
    observation_width: int = Field(default=64, ge=32, le=256)
    frame_stack: int = Field(default=1, ge=1, le=4)
    max_episode_steps: int = Field(default=40, ge=10, le=500)
    action: ActionConfig = Field(default_factory=ActionConfig)


class ReplayConfig(BaseModel):
    capacity: int = Field(default=10_000, ge=128)
    sequence_length: int = Field(default=8, ge=2, le=128)
    batch_size: int = Field(default=4, ge=1, le=512)


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=4, ge=1, le=512)
    imagination_horizon: int = Field(default=5, ge=2, le=64)
    world_model_lr: float = Field(default=3e-4, gt=0.0)
    actor_lr: float = Field(default=8e-5, gt=0.0)
    critic_lr: float = Field(default=8e-5, gt=0.0)
    kl_weight: float = Field(default=1.0, ge=0.0)
    free_nats: float = Field(default=3.0, ge=0.0)
    grad_clip_norm: float = Field(default=100.0, gt=0.0, le=10_000.0)
    use_flash_optimizer: bool = Field(default=False)
    lr_warmup_steps: int = Field(default=0, ge=0, le=10_000)
    world_model_updates_per_cycle: int = Field(default=1, ge=1, le=256)
    behavior_updates_per_cycle: int = Field(default=1, ge=1, le=256)
    cycles: int = Field(default=10, ge=1, le=1_000_000)
    warm_start_steps: int = Field(default=64, ge=0, le=1_000_000)
    policy_steps: int = Field(default=8, ge=0, le=1_000_000)
    checkpoint_interval: int = Field(default=5, ge=1, le=1_000_000)


class ExperimentConfig(BaseModel):
    seed: int = 7
    device: str = "cpu"
    env: EnvConfig = Field(default_factory=EnvConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)
