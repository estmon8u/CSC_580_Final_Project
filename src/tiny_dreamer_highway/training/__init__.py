"""Training utilities for Tiny Dreamer Highway.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from tiny_dreamer_highway.training.behavior_learning import (
	ImaginedTrajectory,
	imagine_trajectory,
	td_lambda_returns,
	train_behavior_step,
)
from tiny_dreamer_highway.training.checkpointing import (
	checkpoint_path,
	find_latest_checkpoint,
	load_checkpoint,
	save_checkpoint,
)
from tiny_dreamer_highway.training.pipeline import (
	PipelineCycleMetrics,
	collect_actor_transitions,
	run_training_cycle,
	seed_latent_state,
)
from tiny_dreamer_highway.training.world_model_step import (
	compute_world_model_losses,
	train_world_model_step,
)
from tiny_dreamer_highway.training.sequence_world_model_step import (
	compute_sequence_world_model_losses,
	stack_sequence_batch,
	train_sequence_world_model_step,
)

__all__ = [
	"ImaginedTrajectory",
	"PipelineCycleMetrics",
	"checkpoint_path",
	"collect_actor_transitions",
	"compute_sequence_world_model_losses",
	"compute_world_model_losses",
	"find_latest_checkpoint",
	"imagine_trajectory",
	"load_checkpoint",
	"run_training_cycle",
	"save_checkpoint",
	"seed_latent_state",
	"stack_sequence_batch",
	"td_lambda_returns",
	"train_behavior_step",
	"train_sequence_world_model_step",
	"train_world_model_step",
]
