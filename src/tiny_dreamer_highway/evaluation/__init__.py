"""Evaluation helpers for Tiny Dreamer Highway.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from tiny_dreamer_highway.evaluation.artifact_bundle import (
	copy_artifact_files,
	create_bundle_archive,
	export_submission_bundle,
	write_bundle_manifest,
)
from tiny_dreamer_highway.evaluation.policy_rollout import (
	DemoBundle,
	RolloutResult,
	record_demo_videos,
	run_policy_episode,
	save_rollout_gif,
)
from tiny_dreamer_highway.evaluation.prediction_eval import (
	compute_frame_metrics,
	evaluate_n_step_predictions,
	rollout_imagined_observations,
)
from tiny_dreamer_highway.evaluation.training_analysis import (
	export_training_history_artifacts,
	load_cycle_metrics_history,
	plot_training_history,
	summarize_training_history,
)
from tiny_dreamer_highway.evaluation.visualization import (
	build_prediction_video_frames,
	export_prediction_artifacts,
	export_prediction_media_bundle,
	export_prediction_video,
	plot_prediction_metrics,
	save_prediction_comparison_grid,
)

__all__ = [
	"DemoBundle",
	"RolloutResult",
	"build_prediction_video_frames",
	"copy_artifact_files",
	"compute_frame_metrics",
	"create_bundle_archive",
	"evaluate_n_step_predictions",
	"export_prediction_artifacts",
	"export_prediction_media_bundle",
	"export_prediction_video",
	"export_submission_bundle",
	"export_training_history_artifacts",
	"load_cycle_metrics_history",
	"plot_training_history",
	"plot_prediction_metrics",
	"record_demo_videos",
	"rollout_imagined_observations",
	"run_policy_episode",
	"save_prediction_comparison_grid",
	"save_rollout_gif",
	"summarize_training_history",
	"write_bundle_manifest",
]