"""Evaluation helpers for Tiny Dreamer Highway.

Name: Esteban
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
from tiny_dreamer_highway.evaluation.prediction_eval import (
	compute_frame_metrics,
	evaluate_n_step_predictions,
	rollout_imagined_observations,
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
	"build_prediction_video_frames",
	"copy_artifact_files",
	"compute_frame_metrics",
	"create_bundle_archive",
	"evaluate_n_step_predictions",
	"export_prediction_artifacts",
	"export_prediction_media_bundle",
	"export_prediction_video",
	"export_submission_bundle",
	"plot_prediction_metrics",
	"rollout_imagined_observations",
	"save_prediction_comparison_grid",
	"write_bundle_manifest",
]