"""End-to-end training runner for Tiny Dreamer Highway.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from tiny_dreamer_highway.config import ExperimentConfig
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.envs.highway_factory import make_highway_env
from tiny_dreamer_highway.models import Actor, Critic, TinyWorldModel
from tiny_dreamer_highway.training.checkpointing import load_checkpoint, save_checkpoint
from tiny_dreamer_highway.training.metrics_logging import export_cycle_metrics, flatten_cycle_metrics
from tiny_dreamer_highway.training.pipeline import PipelineCycleMetrics, run_training_cycle
from tiny_dreamer_highway.utils import set_global_seeds


@dataclass(slots=True)
class TrainingRunSummary:
    total_cycles: int
    completed_cycles: int
    replay_size: int
    latest_record: dict[str, float | int]
    latest_checkpoint: Path | None
    checkpoint_dir: Path
    log_dir: Path


def resolve_training_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def infer_env_shapes(config: ExperimentConfig) -> tuple[tuple[int, int, int], int]:
    env = make_highway_env(config.env)
    try:
        observation, _ = env.reset(seed=config.seed)
        action = env.action_space.sample()
    finally:
        env.close()

    observation_shape = tuple(int(dim) for dim in observation.shape)
    action_dim = int(action.shape[0])
    return observation_shape, action_dim


def initialize_training_state(
    config: ExperimentConfig,
) -> tuple[
    ReplayBuffer,
    TinyWorldModel,
    Actor,
    Critic,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    device = resolve_training_device(config.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    observation_shape, action_dim = infer_env_shapes(config)
    replay_buffer = ReplayBuffer(capacity=config.replay.capacity)
    world_model = TinyWorldModel(observation_shape=observation_shape, action_dim=action_dim).to(device)
    latent_dim = world_model.rssm.deterministic_dim + world_model.rssm.stochastic_dim
    actor = Actor(latent_dim=latent_dim, action_dim=action_dim).to(device)
    critic = Critic(latent_dim=latent_dim).to(device)
    world_model_optimizer = torch.optim.Adam(
        world_model.parameters(),
        lr=config.training.world_model_lr,
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.training.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.training.critic_lr)
    return (
        replay_buffer,
        world_model,
        actor,
        critic,
        world_model_optimizer,
        actor_optimizer,
        critic_optimizer,
    )


def run_training_experiment(
    config: ExperimentConfig,
    artifact_root: str | Path,
    *,
    cycles: int | None = None,
    warm_start_steps: int | None = None,
    policy_steps: int | None = None,
    checkpoint_interval: int | None = None,
    resume_from: str | Path | None = None,
    show_progress: bool = True,
) -> TrainingRunSummary:
    set_global_seeds(config.seed, deterministic_torch=False)

    total_cycles = config.training.cycles if cycles is None else cycles
    initial_warm_start_steps = config.training.warm_start_steps if warm_start_steps is None else warm_start_steps
    cycle_policy_steps = config.training.policy_steps if policy_steps is None else policy_steps
    save_every = config.training.checkpoint_interval if checkpoint_interval is None else checkpoint_interval

    artifact_directory = Path(artifact_root)
    checkpoint_dir = artifact_directory / "checkpoints"
    log_dir = artifact_directory / "logs"

    (
        replay_buffer,
        world_model,
        actor,
        critic,
        world_model_optimizer,
        actor_optimizer,
        critic_optimizer,
    ) = initialize_training_state(config)

    start_step = 1
    if resume_from is not None:
        metadata = load_checkpoint(
            resume_from,
            world_model=world_model,
            actor=actor,
            critic=critic,
            world_model_optimizer=world_model_optimizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            map_location=resolve_training_device(config.device),
        )
        start_step = int(metadata["step"]) + 1

    latest_checkpoint: Path | None = None
    latest_metrics = PipelineCycleMetrics(
        warm_start_added=0,
        policy_added=0,
        replay_size=0,
        world_model_metrics={},
        behavior_metrics={},
    )
    run_start = perf_counter()

    if show_progress:
        print(
            "[train] starting run | "
            f"cycles={total_cycles} | "
            f"start_step={start_step} | "
            f"warm_start_steps={initial_warm_start_steps} | "
            f"policy_steps={cycle_policy_steps} | "
            f"device={resolve_training_device(config.device).type}",
            flush=True,
        )

    for step in range(start_step, total_cycles + 1):
        cycle_start = perf_counter()
        cycle_warm_start_steps = initial_warm_start_steps if step == 1 and start_step == 1 else 0
        latest_metrics = run_training_cycle(
            config,
            replay_buffer,
            world_model,
            actor,
            critic,
            world_model_optimizer,
            actor_optimizer,
            critic_optimizer,
            warm_start_steps=cycle_warm_start_steps,
            policy_steps=cycle_policy_steps,
            seed=config.seed + step - 1,
        )

        checkpoint_file = None
        if step % save_every == 0 or step == total_cycles:
            flattened = flatten_cycle_metrics(step, latest_metrics)
            checkpoint_file = save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                world_model=world_model,
                actor=actor,
                critic=critic,
                world_model_optimizer=world_model_optimizer,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                metrics=flattened,
            )
            latest_checkpoint = checkpoint_file

        export_cycle_metrics(
            log_dir,
            step=step,
            metrics=latest_metrics,
            checkpoint_file=checkpoint_file,
        )

        if show_progress:
            cycle_seconds = perf_counter() - cycle_start
            elapsed_seconds = perf_counter() - run_start
            world_total = latest_metrics.world_model_metrics.get("total_loss")
            actor_loss = latest_metrics.behavior_metrics.get("actor_loss")
            critic_loss = latest_metrics.behavior_metrics.get("critic_loss")
            checkpoint_text = checkpoint_file.name if checkpoint_file is not None else "-"
            print(
                "[train] "
                f"step={step}/{total_cycles} | "
                f"warm={latest_metrics.warm_start_added} | "
                f"policy={latest_metrics.policy_added} | "
                f"replay={latest_metrics.replay_size} | "
                f"world_total={world_total:.4f} | "
                f"actor={actor_loss:.4f} | "
                f"critic={critic_loss:.4f} | "
                f"cycle_s={cycle_seconds:.1f} | "
                f"elapsed_s={elapsed_seconds:.1f} | "
                f"checkpoint={checkpoint_text}",
                flush=True,
            )

    latest_record = flatten_cycle_metrics(total_cycles, latest_metrics)
    return TrainingRunSummary(
        total_cycles=total_cycles,
        completed_cycles=total_cycles,
        replay_size=len(replay_buffer),
        latest_record=latest_record,
        latest_checkpoint=latest_checkpoint,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )