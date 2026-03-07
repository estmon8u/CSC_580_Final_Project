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


class RewardConfig(BaseModel):
    collision_reward: float = -1.0
    right_lane_reward: float = 0.1
    high_speed_reward: float = 0.4
    lane_change_reward: float = 0.0
    normalize_reward: bool = True
    reward_speed_range: tuple[float, float] = (20.0, 30.0)
    offroad_terminal: bool = True
    offroad_penalty: float = Field(default=3.0, ge=0.0)
    steering_penalty: float = Field(default=0.05, ge=0.0)
    steering_change_penalty: float = Field(default=0.1, ge=0.0)


class EnvConfig(BaseModel):
    env_id: str = "highway-v0"
    observation_height: int = Field(default=64, ge=32, le=256)
    observation_width: int = Field(default=64, ge=32, le=256)
    frame_stack: int = Field(default=1, ge=1, le=4)
    max_episode_steps: int = Field(default=40, ge=10, le=500)
    action: ActionConfig = Field(default_factory=ActionConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)


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
    lr_warmup_steps: int = Field(default=0, ge=0, le=10_000)
    use_amp: bool = False
    amp_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    use_flash_optimizer: bool = False
    world_model_updates_per_cycle: int = Field(default=1, ge=1, le=256)
    behavior_updates_per_cycle: int = Field(default=1, ge=1, le=256)
    cycles: int = Field(default=10, ge=1, le=1_000_000)
    warm_start_steps: int = Field(default=64, ge=0, le=1_000_000)
    policy_steps: int = Field(default=8, ge=0, le=1_000_000)
    checkpoint_interval: int = Field(default=5, ge=1, le=1_000_000)


class ModelConfig(BaseModel):
    """Configurable model dimensions — matches DreamerV1 reference defaults."""

    embedding_dim: int = Field(default=1024, ge=32, le=4096)
    deterministic_dim: int = Field(default=200, ge=32, le=2048)
    stochastic_dim: int = Field(default=30, ge=8, le=512)
    hidden_dim: int = Field(default=200, ge=32, le=2048)
    rssm_num_layers: int = Field(default=2, ge=1, le=4)
    actor_hidden_dim: int = Field(default=200, ge=32, le=2048)
    actor_num_layers: int = Field(default=2, ge=1, le=4)
    critic_hidden_dim: int = Field(default=200, ge=32, le=2048)
    critic_num_layers: int = Field(default=3, ge=1, le=6)
    reward_hidden_dim: int = Field(default=200, ge=32, le=2048)
    reward_num_layers: int = Field(default=2, ge=1, le=4)


class ExperimentConfig(BaseModel):
    seed: int = 7
    device: str = "cpu"
    env: EnvConfig = Field(default_factory=EnvConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)
