"""End-to-end training runner for Tiny Dreamer Highway.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil
from time import perf_counter

import torch

from tiny_dreamer_highway.config import ExperimentConfig
from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
from tiny_dreamer_highway.envs.highway_factory import make_highway_env
from tiny_dreamer_highway.models import Actor, Critic, TinyWorldModel
from tiny_dreamer_highway.training.checkpointing import load_checkpoint, save_checkpoint
from tiny_dreamer_highway.training.metrics_logging import export_cycle_metrics, flatten_cycle_metrics
from tiny_dreamer_highway.training.pipeline import PipelineCycleMetrics, resolve_amp_dtype, run_training_cycle
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


def evaluate_training_policy(
    config: ExperimentConfig,
    world_model: TinyWorldModel,
    actor: Actor,
    *,
    episodes: int,
    max_steps: int,
    seed: int | None = None,
) -> dict[str, float]:
    if episodes <= 0:
        return {}

    from tiny_dreamer_highway.evaluation.policy_rollout import run_policy_episode

    world_model_was_training = world_model.training
    actor_was_training = actor.training
    world_model.eval()
    actor.eval()

    rewards: list[float] = []
    steps: list[int] = []
    crashes = 0
    try:
        for episode_index in range(episodes):
            episode_seed = None if seed is None else seed + episode_index
            result = run_policy_episode(
                config,
                world_model,
                actor,
                max_steps=max_steps,
                seed=episode_seed,
                capture_frames=False,
            )
            rewards.append(result.total_reward)
            steps.append(result.steps)
            crashes += int(result.terminated)
    finally:
        world_model.train(world_model_was_training)
        actor.train(actor_was_training)

    return {
        "episodes": float(episodes),
        "mean_reward": float(sum(rewards) / len(rewards)),
        "mean_steps": float(sum(steps) / len(steps)),
        "crash_rate": float(crashes / episodes),
    }


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


def _try_flash_adamw(
    params,
    lr: float,
) -> torch.optim.Optimizer | None:
    """Try to create a FlashAdamW optimizer (Linux + CUDA only).

    Returns *None* when the package is not installed or the platform
    is unsupported, so callers can fall back to standard AdamW.
    """
    try:
        from flashoptim import FlashAdamW  # type: ignore[import-untyped]
        optimizer = FlashAdamW(params, lr=lr)
        print("[optimizer] using FlashAdamW", flush=True)
        return optimizer
    except ImportError:
        print("[optimizer] flashoptim not installed — falling back to AdamW", flush=True)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[optimizer] FlashAdamW failed ({exc!r}) — falling back to AdamW", flush=True)
        return None


def _make_optimizer(
    params,
    lr: float,
    *,
    use_flash: bool = False,
) -> torch.optim.Optimizer:
    """Create an optimizer — FlashAdamW when requested and available, else AdamW."""
    # Materialise the generator so it can survive a failed FlashAdamW attempt.
    params = list(params)
    if use_flash:
        optimizer = _try_flash_adamw(params, lr)
        if optimizer is not None:
            return optimizer
    return torch.optim.AdamW(params, lr=lr)


def _make_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Linear warm-up from ~0 to base LR over *warmup_steps* optimizer steps."""
    if warmup_steps <= 0:
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
    )


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
        observation_distribution_std=mc.observation_distribution_std,
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

    _flash = config.training.use_flash_optimizer and device.type == "cuda"

    # FlashAdamW requires model weights in half precision — cast when both
    # AMP *and* flash are requested so the fused optimizer can manage master
    # weights internally.
    if _flash and config.training.use_amp:
        _dtype = resolve_amp_dtype(config.training.amp_dtype)
        world_model = world_model.to(_dtype)
        actor = actor.to(_dtype)
        critic = critic.to(_dtype)
        print(
            f"[optimizer] cast models to {config.training.amp_dtype} for FlashAdamW",
            flush=True,
        )

    world_model_optimizer = _make_optimizer(
        world_model.parameters(),
        lr=config.training.world_model_lr,
        use_flash=_flash,
    )
    actor_optimizer = _make_optimizer(actor.parameters(), lr=config.training.actor_lr, use_flash=_flash)
    critic_optimizer = _make_optimizer(critic.parameters(), lr=config.training.critic_lr, use_flash=_flash)
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
    if resume_from is None and artifact_directory.exists():
        shutil.rmtree(artifact_directory)
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

    # LR warm-up schedulers (None when warmup_steps == 0)
    wm_scheduler = _make_warmup_scheduler(world_model_optimizer, config.training.lr_warmup_steps)
    actor_scheduler = _make_warmup_scheduler(actor_optimizer, config.training.lr_warmup_steps)
    critic_scheduler = _make_warmup_scheduler(critic_optimizer, config.training.lr_warmup_steps)

    # AMP – automatic mixed precision (disabled by default)
    device = resolve_training_device(config.device)
    if config.training.use_amp and device.type == "cuda":
        amp_dtype = resolve_amp_dtype(config.training.amp_dtype)
        amp_context = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        # bfloat16 does not need scaling; float16 does
        _use_scaler = amp_dtype == torch.float16
        wm_scaler = torch.amp.GradScaler("cuda") if _use_scaler else None
        actor_scaler = torch.amp.GradScaler("cuda") if _use_scaler else None
        critic_scaler = torch.amp.GradScaler("cuda") if _use_scaler else None
    else:
        amp_context = None
        wm_scaler = None
        actor_scaler = None
        critic_scaler = None

    latest_checkpoint: Path | None = None
    latest_metrics = PipelineCycleMetrics(
        warm_start_added=0,
        policy_added=0,
        replay_size=0,
        world_model_metrics={},
        behavior_metrics={},
        evaluation_metrics={},
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
            wm_scaler=wm_scaler,
            actor_scaler=actor_scaler,
            critic_scaler=critic_scaler,
            amp_context=amp_context,
        )

        eval_cfg = config.evaluation
        if eval_cfg.episodes > 0 and eval_cfg.interval > 0 and step % eval_cfg.interval == 0:
            latest_metrics.evaluation_metrics = evaluate_training_policy(
                config,
                world_model,
                actor,
                episodes=eval_cfg.episodes,
                max_steps=eval_cfg.max_steps,
                seed=config.seed + step * 1_000,
            )

        # Step LR warm-up schedulers (no-op when scheduler is None)
        for scheduler in (wm_scheduler, actor_scheduler, critic_scheduler):
            if scheduler is not None:
                scheduler.step()

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
            eval_reward = latest_metrics.evaluation_metrics.get("mean_reward")
            checkpoint_text = checkpoint_file.name if checkpoint_file is not None else "-"
            wt_str = f"{world_total:.4f}" if world_total is not None else "n/a"
            al_str = f"{actor_loss:.4f}" if actor_loss is not None else "n/a"
            cl_str = f"{critic_loss:.4f}" if critic_loss is not None else "n/a"
            er_str = f"{eval_reward:.4f}" if eval_reward is not None else "n/a"
            print(
                "[train] "
                f"step={step}/{total_cycles} | "
                f"warm={latest_metrics.warm_start_added} | "
                f"policy={latest_metrics.policy_added} | "
                f"replay={latest_metrics.replay_size} | "
                f"world_total={wt_str} | "
                f"actor={al_str} | "
                f"critic={cl_str} | "
                f"eval_reward={er_str} | "
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