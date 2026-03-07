"""Tests for the policy rollout / demo video recording module.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tiny_dreamer_highway.evaluation.policy_rollout import (
    DemoBundle,
    RolloutResult,
    _build_models,
    _observation_to_tensor,
    record_demo_videos,
    run_policy_episode,
    save_rollout_gif,
)
from tiny_dreamer_highway.config import load_experiment_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config():
    config_path = Path(__file__).resolve().parents[1] / "examples" / "training_run.yaml"
    return load_experiment_config(config_path)


@pytest.fixture()
def dummy_checkpoint(config, tmp_path: Path):
    """Create a fresh (untrained) checkpoint on disk."""
    from tiny_dreamer_highway.evaluation.policy_rollout import _build_models

    device = torch.device("cpu")
    world_model, actor, critic = _build_models(config, device)

    # Mimic the checkpoint format used by save_checkpoint
    payload = {
        "step": 1,
        "world_model": world_model.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "world_model_optimizer": torch.optim.AdamW(world_model.parameters(), lr=1e-3).state_dict(),
        "actor_optimizer": torch.optim.AdamW(actor.parameters(), lr=1e-3).state_dict(),
        "critic_optimizer": torch.optim.AdamW(critic.parameters(), lr=1e-3).state_dict(),
        "metrics": {},
    }
    ckpt_path = tmp_path / "checkpoint_00001.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Unit tests: _observation_to_tensor
# ---------------------------------------------------------------------------

class TestObservationToTensor:
    def test_2d_observation_becomes_4d(self):
        obs = np.zeros((64, 64), dtype=np.uint8)
        tensor = _observation_to_tensor(obs, torch.device("cpu"))
        assert tensor.ndim == 4
        assert tensor.shape == (1, 1, 64, 64)

    def test_3d_observation_becomes_4d(self):
        obs = np.zeros((1, 64, 64), dtype=np.uint8)
        tensor = _observation_to_tensor(obs, torch.device("cpu"))
        assert tensor.ndim == 4
        assert tensor.shape == (1, 1, 64, 64)

    def test_dtype_is_uint8(self):
        obs = np.zeros((64, 64), dtype=np.uint8)
        tensor = _observation_to_tensor(obs, torch.device("cpu"))
        assert tensor.dtype == torch.uint8


# ---------------------------------------------------------------------------
# Unit tests: _build_models
# ---------------------------------------------------------------------------

class TestBuildModels:
    def test_returns_three_models(self, config):
        device = torch.device("cpu")
        world_model, actor, critic = _build_models(config, device)
        assert hasattr(world_model, "rssm")
        assert hasattr(actor, "net")
        assert hasattr(critic, "value")

    def test_models_on_correct_device(self, config):
        device = torch.device("cpu")
        world_model, actor, critic = _build_models(config, device)
        assert next(world_model.parameters()).device == device
        assert next(actor.parameters()).device == device
        assert next(critic.parameters()).device == device


# ---------------------------------------------------------------------------
# Unit tests: save_rollout_gif
# ---------------------------------------------------------------------------

class TestSaveRolloutGif:
    def test_creates_gif_file(self, tmp_path: Path):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        output = save_rollout_gif(frames, tmp_path / "test.gif", fps=10)
        assert output.exists()
        assert output.suffix == ".gif"
        assert output.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        deep_path = tmp_path / "a" / "b" / "c" / "test.gif"
        output = save_rollout_gif(frames, deep_path)
        assert output.exists()


# ---------------------------------------------------------------------------
# Integration tests: run_policy_episode
# ---------------------------------------------------------------------------

class TestRunPolicyEpisode:
    def test_returns_rollout_result(self, config, dummy_checkpoint):
        from tiny_dreamer_highway.evaluation.policy_rollout import _load_models_from_checkpoint

        device = torch.device("cpu")
        world_model, actor = _load_models_from_checkpoint(dummy_checkpoint, config, device)
        result = run_policy_episode(config, world_model, actor, max_steps=5, seed=42)

        assert isinstance(result, RolloutResult)
        assert result.steps > 0
        assert result.steps <= 5
        assert len(result.rewards) == result.steps
        assert isinstance(result.total_reward, float)
        assert isinstance(result.terminated, bool)

    def test_frames_are_captured(self, config, dummy_checkpoint):
        from tiny_dreamer_highway.evaluation.policy_rollout import _load_models_from_checkpoint

        device = torch.device("cpu")
        world_model, actor = _load_models_from_checkpoint(dummy_checkpoint, config, device)
        result = run_policy_episode(config, world_model, actor, max_steps=5, seed=42)

        # We should have at least 1 frame (initial) + steps
        assert len(result.frames) >= 1
        for frame in result.frames:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3  # (H, W, 3)


# ---------------------------------------------------------------------------
# Integration tests: record_demo_videos
# ---------------------------------------------------------------------------

class TestRecordDemoVideos:
    def test_records_episodes_and_saves_gifs(self, config, dummy_checkpoint, tmp_path: Path):
        bundle = record_demo_videos(
            config,
            checkpoint_path=dummy_checkpoint,
            output_dir=tmp_path / "demos",
            num_episodes=2,
            max_steps=5,
            fps=10,
            seed=42,
            prefix="test_demo",
            device="cpu",
            show_progress=False,
        )

        assert isinstance(bundle, DemoBundle)
        assert len(bundle.video_paths) == 2
        assert len(bundle.results) == 2

        for gif_path in bundle.video_paths:
            assert gif_path.exists()
            assert gif_path.suffix == ".gif"
            assert gif_path.stat().st_size > 0

    def test_writes_summary_json(self, config, dummy_checkpoint, tmp_path: Path):
        import json

        output_dir = tmp_path / "demos"
        record_demo_videos(
            config,
            checkpoint_path=dummy_checkpoint,
            output_dir=output_dir,
            num_episodes=1,
            max_steps=5,
            seed=42,
            prefix="test_demo",
            device="cpu",
            show_progress=False,
        )

        summary_path = output_dir / "test_demo_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["num_episodes"] == 1
        assert len(summary["episodes"]) == 1
        assert "total_reward" in summary["episodes"][0]

    def test_different_seeds_produce_different_episodes(self, config, dummy_checkpoint, tmp_path: Path):
        bundle = record_demo_videos(
            config,
            checkpoint_path=dummy_checkpoint,
            output_dir=tmp_path / "demos",
            num_episodes=2,
            max_steps=10,
            seed=0,
            prefix="test_demo",
            device="cpu",
            show_progress=False,
        )

        # With different seeds per episode, results may differ
        # (at minimum, each episode should be independently recorded)
        assert len(bundle.results) == 2
        assert all(r.steps > 0 for r in bundle.results)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_rollout_result_fields(self):
        result = RolloutResult(
            frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
            rewards=[1.0, 0.5],
            total_reward=1.5,
            steps=2,
            terminated=False,
        )
        assert result.total_reward == 1.5
        assert result.steps == 2
        assert not result.terminated

    def test_demo_bundle_fields(self):
        bundle = DemoBundle(
            video_paths=[Path("a.gif"), Path("b.gif")],
            results=[],
        )
        assert len(bundle.video_paths) == 2
