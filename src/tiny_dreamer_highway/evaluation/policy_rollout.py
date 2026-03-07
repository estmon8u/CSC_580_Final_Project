"""Record agent-driving demo videos from a trained policy.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot

This module loads a trained checkpoint and runs the actor policy inside the
real highway-env, capturing rendered RGB frames and saving them as animated
GIFs that show the agent actually driving.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor

from tiny_dreamer_highway.config import ExperimentConfig, load_experiment_config
from tiny_dreamer_highway.envs.highway_factory import make_highway_env
from tiny_dreamer_highway.models import Actor, Critic, LatentState, TinyWorldModel
from tiny_dreamer_highway.training.checkpointing import find_latest_checkpoint
from tiny_dreamer_highway.training.experiment import (
    infer_env_shapes,
    resolve_training_device,
)
from tiny_dreamer_highway.utils import stabilize_action_tensor


@dataclass(slots=True)
class RolloutResult:
    """Holds the outputs of a single policy rollout episode."""

    frames: list[np.ndarray]
    rewards: list[float]
    total_reward: float
    steps: int
    terminated: bool


@dataclass(slots=True)
class DemoBundle:
    """Paths produced by :func:`record_demo_videos`."""

    video_paths: list[Path]
    results: list[RolloutResult]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _observation_to_tensor(observation: np.ndarray, device: torch.device) -> Tensor:
    """Convert a single numpy observation to a batched tensor on *device*."""
    tensor = torch.as_tensor(np.asarray(observation, dtype=np.uint8), device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _build_models(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[TinyWorldModel, Actor, Critic]:
    """Instantiate world-model, actor, and critic on *device* (weights uninitialized)."""
    observation_shape, action_dim = infer_env_shapes(config)
    mc = config.model
    world_model = TinyWorldModel(
        observation_shape=observation_shape,
        action_dim=action_dim,
        embedding_dim=mc.embedding_dim,
        deterministic_dim=mc.deterministic_dim,
        stochastic_dim=mc.stochastic_dim,
        hidden_dim=mc.hidden_dim,
        rssm_min_std=mc.rssm_min_std,
        rssm_num_layers=mc.rssm_num_layers,
        reward_hidden_dim=mc.reward_hidden_dim,
        reward_num_layers=mc.reward_num_layers,
        reward_distribution_std=mc.reward_distribution_std,
        use_continue_model=mc.use_continue_model,
        continue_hidden_dim=mc.continue_hidden_dim,
        continue_num_layers=mc.continue_num_layers,
    ).to(device)
    latent_dim = world_model.rssm.deterministic_dim + world_model.rssm.stochastic_dim
    actor = Actor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=mc.actor_hidden_dim,
        num_layers=mc.actor_num_layers,
        init_std=mc.actor_init_std,
        mean_scale=mc.actor_mean_scale,
        min_std=mc.actor_min_std,
    ).to(device)
    critic = Critic(
        latent_dim=latent_dim,
        hidden_dim=mc.critic_hidden_dim,
        num_layers=mc.critic_num_layers,
        distribution_std=mc.critic_distribution_std,
    ).to(device)
    return world_model, actor, critic


def _load_models_from_checkpoint(
    checkpoint_path: str | Path,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[TinyWorldModel, Actor]:
    """Load world-model and actor weights from a saved checkpoint."""
    world_model, actor, _critic = _build_models(config, device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    world_model.load_state_dict(checkpoint["world_model"])
    actor.load_state_dict(checkpoint["actor"])
    world_model.eval()
    actor.eval()
    return world_model, actor


# ------------------------------------------------------------------
# Core rollout
# ------------------------------------------------------------------

def run_policy_episode(
    config: ExperimentConfig,
    world_model: TinyWorldModel,
    actor: Actor,
    *,
    max_steps: int = 200,
    seed: int | None = None,
    capture_frames: bool = True,
) -> RolloutResult:
    """Run the trained actor in the real highway-env for one episode.

    At each step the encoder + RSSM posterior provide a latent state,
    the actor selects an action, and the environment is stepped.  The
    rendered RGB frame from ``env.render()`` is captured every step.

    Parameters
    ----------
    config:
        Experiment config (used to build the environment).
    world_model:
        Trained ``TinyWorldModel`` (encoder + RSSM).
    actor:
        Trained ``Actor`` network.
    max_steps:
        Hard cap on episode length.
    seed:
        Optional environment seed for reproducibility.

    Returns
    -------
    RolloutResult
        Frames, per-step rewards, total reward, step count, and
        whether the episode ended via termination (crash) or
        truncation (time limit).
    """
    env = make_highway_env(config.env)
    device = next(world_model.parameters()).device
    action_dim = world_model.rssm.action_dim
    observation, _ = env.reset(seed=seed)
    prev_state = world_model.rssm.initial_state(batch_size=1, device=device)
    prev_action = torch.zeros(1, action_dim, device=device)

    frames: list[np.ndarray] = []
    rewards: list[float] = []
    terminated = False

    try:
        # capture the initial rendered frame
        if capture_frames:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

        for _ in range(max_steps):
            with torch.no_grad():
                # 1. Encode current observation → posterior (uses previous action for GRU)
                obs_tensor = _observation_to_tensor(observation, device)
                output = world_model(obs_tensor, prev_action, prev_state=prev_state)
                prev_state = output.posterior_state

                # 2. Select action based on posterior that sees current obs
                action_tensor = stabilize_action_tensor(
                    actor(prev_state.features),
                    previous_action=prev_action,
                    longitudinal_scale=config.env.action.longitudinal_scale,
                    lateral_scale=config.env.action.lateral_scale,
                    smoothing_factor=config.env.action.smoothing_factor,
                    lateral_enabled=config.env.action.lateral,
                )
                prev_action = action_tensor

            action = action_tensor.squeeze(0).float().cpu().numpy()
            next_observation, reward, term, trunc, _ = env.step(action)
            rewards.append(float(reward))

            if capture_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

            observation = next_observation
            if term or trunc:
                terminated = bool(term)
                break
    finally:
        env.close()

    return RolloutResult(
        frames=frames,
        rewards=rewards,
        total_reward=sum(rewards),
        steps=len(rewards),
        terminated=terminated,
    )


# ------------------------------------------------------------------
# GIF export
# ------------------------------------------------------------------

def save_rollout_gif(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    fps: int = 15,
) -> Path:
    """Write a sequence of RGB frames to an animated GIF.

    Parameters
    ----------
    frames:
        List of ``(H, W, 3)`` uint8 numpy arrays.
    output_path:
        Destination file path (will be created / overwritten).
    fps:
        Frames per second for the GIF.

    Returns
    -------
    Path
        The resolved output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(str(path), list(frames), duration=duration, loop=0)
    return path


# ------------------------------------------------------------------
# High-level convenience
# ------------------------------------------------------------------

def record_demo_videos(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    num_episodes: int = 3,
    max_steps: int = 200,
    fps: int = 15,
    seed: int | None = None,
    prefix: str = "demo",
    device: str = "cpu",
    show_progress: bool = True,
) -> DemoBundle:
    """Record *num_episodes* policy rollouts and save as animated GIFs.

    This is the main entry point for producing agent-driving demo
    videos.  It loads the checkpoint, runs the actor in the real
    environment, and writes one GIF per episode plus a summary JSON.

    Parameters
    ----------
    config:
        Experiment configuration.
    checkpoint_path:
        Path to a saved ``.pt`` checkpoint file.
    output_dir:
        Directory where GIFs (and an optional summary) are written.
    num_episodes:
        How many episodes to record.
    max_steps:
        Maximum steps per episode.
    fps:
        GIF frame rate.
    seed:
        Base seed (incremented per episode for variety).
    prefix:
        Filename prefix, e.g. ``"demo"`` → ``demo_ep01.gif``.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    show_progress:
        Print a line per completed episode.

    Returns
    -------
    DemoBundle
        Paths to the saved GIFs and per-episode statistics.
    """
    torch_device = resolve_training_device(device)
    world_model, actor = _load_models_from_checkpoint(checkpoint_path, config, torch_device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    video_paths: list[Path] = []
    results: list[RolloutResult] = []

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        result = run_policy_episode(
            config,
            world_model,
            actor,
            max_steps=max_steps,
            seed=ep_seed,
        )
        results.append(result)

        gif_path = save_rollout_gif(
            result.frames,
            out / f"{prefix}_ep{ep + 1:02d}.gif",
            fps=fps,
        )
        video_paths.append(gif_path)

        if show_progress:
            tag = "CRASH" if result.terminated else "OK"
            print(
                f"[demo] episode {ep + 1}/{num_episodes} | "
                f"steps={result.steps} | reward={result.total_reward:.2f} | "
                f"{tag} | saved {gif_path.name}",
                flush=True,
            )

    # Write a lightweight JSON summary alongside the GIFs
    import json

    summary = {
        "num_episodes": num_episodes,
        "checkpoint": str(checkpoint_path),
        "episodes": [
            {
                "episode": i + 1,
                "steps": r.steps,
                "total_reward": r.total_reward,
                "terminated": r.terminated,
                "gif": str(video_paths[i].name),
            }
            for i, r in enumerate(results)
        ],
    }
    summary_path = out / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if show_progress:
        avg_reward = sum(r.total_reward for r in results) / max(len(results), 1)
        print(
            f"[demo] done | avg_reward={avg_reward:.2f} | "
            f"summary={summary_path.name}",
            flush=True,
        )

    return DemoBundle(video_paths=video_paths, results=results)
