"""Training utilities for Tiny Dreamer Highway.

Name: Esteban Montelongo
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
from tiny_dreamer_highway.training.metrics_logging import (
	append_metrics_csv,
	append_metrics_jsonl,
	export_cycle_metrics,
	flatten_cycle_metrics,
	write_artifact_summary,
)
from tiny_dreamer_highway.training.pipeline import (
	PipelineCycleMetrics,
	collect_actor_transitions,
	run_training_cycle,
	seed_latent_state,
)
from tiny_dreamer_highway.training.experiment import (
	TrainingRunSummary,
	initialize_training_state,
	run_training_experiment,
	resolve_training_device,
)
from tiny_dreamer_highway.training.world_model_step import (
	compute_world_model_losses,
	gaussian_kl_divergence,
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
	"TrainingRunSummary",
	"checkpoint_path",
	"collect_actor_transitions",
	"compute_sequence_world_model_losses",
	"compute_world_model_losses",
	"gaussian_kl_divergence",
	"append_metrics_csv",
	"append_metrics_jsonl",
	"export_cycle_metrics",
	"find_latest_checkpoint",
	"flatten_cycle_metrics",
	"imagine_trajectory",
	"initialize_training_state",
	"load_checkpoint",
	"resolve_training_device",
	"run_training_cycle",
	"run_training_experiment",
	"save_checkpoint",
	"seed_latent_state",
	"stack_sequence_batch",
	"td_lambda_returns",
	"train_behavior_step",
	"train_sequence_world_model_step",
	"train_world_model_step",
	"write_artifact_summary",
]
