"""Typed experiment configuration for Tiny Dreamer Highway.

All hyperparameters — environment, replay buffer, training schedule,
model dimensions, and evaluation settings — are defined as Pydantic
models with validation constraints.  A YAML file is loaded via
``load_experiment_config()`` and validated into an ``ExperimentConfig``.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ActionConfig(BaseModel):
    type: Literal["continuous", "discrete"] = "continuous"
    longitudinal: bool = True
    lateral: bool = True
    longitudinal_scale: float = Field(default=1.0, gt=0.0, le=1.0)
    lateral_scale: float = Field(default=0.35, gt=0.0, le=1.0)
    smoothing_factor: float = Field(default=0.6, ge=0.0, lt=1.0)
    num_actions: int = Field(default=5, ge=2, le=20)

    @property
    def is_discrete(self) -> bool:
        return self.type == "discrete"


class RewardConfig(BaseModel):
    collision_reward: float = -1.0
    right_lane_reward: float = 0.1
    high_speed_reward: float = 0.4
    lane_change_reward: float = 0.0
    overtake_reward: float = Field(default=0.0, ge=0.0)
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
    lanes_count: int = Field(default=4, ge=2, le=8)
    vehicles_count: int = Field(default=50, ge=0, le=200)
    npc_speed_scale: float = Field(default=1.0, gt=0.5, le=1.0)
    simulation_frequency: int = Field(default=15, ge=1, le=60)
    policy_frequency: int = Field(default=5, ge=1, le=30)
    action: ActionConfig = Field(default_factory=ActionConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="after")
    def _validate_frequency_relationship(self) -> "EnvConfig":
        if self.simulation_frequency < self.policy_frequency:
            raise ValueError(
                f"simulation_frequency ({self.simulation_frequency}) must be >= "
                f"policy_frequency ({self.policy_frequency}); the simulation "
                "must tick at least as fast as the policy."
            )
        return self


class ReplayConfig(BaseModel):
    capacity: int = Field(default=10_000, ge=128)
    sequence_length: int = Field(default=8, ge=2, le=128)
    batch_size: int = Field(default=4, ge=1, le=512)


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=4, ge=1, le=512)
    imagination_horizon: int = Field(default=5, ge=2, le=64)
    discount: float = Field(default=0.99, ge=0.0, le=1.0)
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0)
    overshooting_horizon: int = Field(default=3, ge=0, le=32)
    overshooting_kl_weight: float = Field(default=0.5, ge=0.0, le=10.0)
    world_model_lr: float = Field(default=3e-4, gt=0.0)
    actor_lr: float = Field(default=8e-5, gt=0.0)
    critic_lr: float = Field(default=8e-5, gt=0.0)
    kl_weight: float = Field(default=1.0, ge=0.0)
    kl_balance: float = Field(default=0.8, ge=0.0, le=1.0)
    free_nats: float = Field(default=3.0, ge=0.0)
    continue_loss_weight: float = Field(default=1.0, ge=0.0)
    grad_clip_norm: float = Field(default=100.0, gt=0.0, le=10_000.0)
    lr_warmup_steps: int = Field(default=0, ge=0, le=10_000)
    use_amp: bool = False
    amp_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    deterministic_torch: bool = True
    actor_entropy_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    world_model_updates_per_cycle: int = Field(default=1, ge=1, le=256)
    behavior_updates_per_cycle: int = Field(default=1, ge=1, le=256)
    cycles: int = Field(default=10, ge=1, le=1_000_000)
    warm_start_steps: int = Field(default=64, ge=0, le=1_000_000)
    policy_steps: int = Field(default=8, ge=0, le=1_000_000)
    checkpoint_interval: int = Field(default=5, ge=1, le=1_000_000)


class EvaluationConfig(BaseModel):
    interval: int = Field(default=0, ge=0, le=1_000_000)
    episodes: int = Field(default=0, ge=0, le=1_000)
    max_steps: int = Field(default=200, ge=1, le=10_000)


class ModelConfig(BaseModel):
    """Configurable model dimensions — DreamerV2 categorical latent state."""

    embedding_dim: int = Field(default=1024, ge=32, le=4096)
    deterministic_dim: int = Field(default=200, ge=32, le=2048)
    num_categoricals: int = Field(default=32, ge=1, le=128)
    num_classes: int = Field(default=32, ge=2, le=128)
    hidden_dim: int = Field(default=200, ge=32, le=2048)
    rssm_num_layers: int = Field(default=2, ge=1, le=4)
    rssm_min_std: float = Field(default=0.1, gt=0.0, le=5.0)

    @property
    def stochastic_dim(self) -> int:
        """Flattened stochastic state size = num_categoricals * num_classes."""
        return self.num_categoricals * self.num_classes
    actor_hidden_dim: int = Field(default=200, ge=32, le=2048)
    actor_num_layers: int = Field(default=2, ge=1, le=4)
    actor_init_std: float = Field(default=5.0, gt=0.0, le=10.0)
    actor_mean_scale: float = Field(default=5.0, gt=0.0, le=10.0)
    actor_min_std: float = Field(default=1e-4, gt=0.0, le=1.0)
    critic_hidden_dim: int = Field(default=200, ge=32, le=2048)
    critic_num_layers: int = Field(default=3, ge=1, le=6)
    critic_distribution_std: float = Field(default=1.0, gt=0.0, le=10.0)
    observation_distribution_std: float = Field(default=1.0, gt=0.0, le=10.0)
    reward_hidden_dim: int = Field(default=200, ge=32, le=2048)
    reward_num_layers: int = Field(default=2, ge=1, le=4)
    reward_distribution_std: float = Field(default=1.0, gt=0.0, le=10.0)
    use_continue_model: bool = True
    continue_hidden_dim: int = Field(default=200, ge=32, le=2048)
    continue_num_layers: int = Field(default=2, ge=1, le=4)


class ExperimentConfig(BaseModel):
    seed: int = 7
    device: str = "cpu"
    env: EnvConfig = Field(default_factory=EnvConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)
