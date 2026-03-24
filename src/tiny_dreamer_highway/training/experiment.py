"""End-to-end training runner for the Tiny Dreamer Highway agent.

This module is the top-level entry point for a training experiment.  It
wires together all components: environment creation, model initialization,
optimizer setup, checkpoint resume, the alternating train–collect loop,
periodic evaluation, metric export, and artifact management.

Experiment lifecycle:

1. **Initialization** — infer observation/action shapes from a probe
   environment, instantiate the world model, actor, critic, optimizers,
   and replay buffer.
2. **Optional resume** — reload model/optimizer/scheduler/replay state
   from a checkpoint file.  If the checkpoint is missing, the run
   starts fresh and old artifacts are cleared.
3. **Training loop** — iterate ``total_cycles`` training cycles (see
   ``pipeline.run_training_cycle``).  Each cycle alternates world-model
   updates, behavior updates, and real-environment policy collection.
4. **Periodic evaluation** — run deterministic policy episodes and log
   mean reward, crash rate, and episode length.
5. **Checkpointing** — save full training state (model weights,
   optimizer state, replay buffer, LR scheduler state) at configured
   intervals.
6. **Logging** — export per-cycle metrics to the ``logs/`` directory.

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
from tiny_dreamer_highway.models import Actor, Critic, DiscreteActor, TinyWorldModel
from tiny_dreamer_highway.training.checkpointing import load_checkpoint, save_checkpoint
from tiny_dreamer_highway.training.metrics_logging import export_cycle_metrics, flatten_cycle_metrics
from tiny_dreamer_highway.training.pipeline import PipelineCycleMetrics, resolve_amp_dtype, run_training_cycle
from tiny_dreamer_highway.utils import set_global_seeds


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _fmt(value: float | None, width: int = 10, precision: int = 4) -> str:
    if value is None:
        return "n/a".rjust(width)
    if abs(value) < 0.01 and value != 0.0:
        return f"{value:{width}.3e}"
    return f"{value:{width}.{precision}f}"


def _print_debug_cycle_details(
    step: int,
    total_cycles: int,
    metrics: PipelineCycleMetrics,
    *,
    world_model_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    checkpoint_file: Path | None,
) -> None:
    wm = metrics.world_model_metrics
    beh = metrics.behavior_metrics
    ev = metrics.evaluation_metrics
    warnings: list[str] = []

    sep = "─" * 72
    print(f"\n[DEBUG] {'═' * 20} CYCLE {step}/{total_cycles} DIAGNOSTIC {'═' * 20}", flush=True)

    # ── Learning rates ───────────────────────────────────────────────
    wm_lr = _optimizer_lr(world_model_optimizer)
    actor_lr = _optimizer_lr(actor_optimizer)
    critic_lr = _optimizer_lr(critic_optimizer)
    print(f"  LR  │ world_model={wm_lr:.2e}  actor={actor_lr:.2e}  critic={critic_lr:.2e}")
    print(f"  DATA│ replay={metrics.replay_size}  warm_added={metrics.warm_start_added}  policy_added={metrics.policy_added}")

    # ── World model ──────────────────────────────────────────────────
    print(f"\n  {sep}")
    print("  WORLD MODEL")
    print(f"  {sep}")
    recon = wm.get("reconstruction_loss")
    rew_loss = wm.get("reward_loss")
    kl = wm.get("kl_loss")
    kl_raw = wm.get("kl_loss_raw")
    kl_dyn = wm.get("kl_dynamics")
    kl_rep = wm.get("kl_representation")
    cont = wm.get("continue_loss")
    overshoot_kl = wm.get("overshooting_kl_loss")
    total = wm.get("total_loss")
    wm_gn = wm.get("wm_grad_norm")

    print(f"    reconstruction_loss : {_fmt(recon)}")
    print(f"    reward_loss         : {_fmt(rew_loss)}")
    print(f"    kl_loss (weighted)  : {_fmt(kl)}    kl_raw: {_fmt(kl_raw)}")
    print(f"    kl_dynamics         : {_fmt(kl_dyn)}    kl_representation: {_fmt(kl_rep)}")
    # NOTE: kl_dynamics == kl_representation is expected by design.
    # They compute the same KL(posterior || prior); the α-balance happens
    # via stop-gradients that route 80% of gradient to the prior and 20%
    # to the posterior — the raw scalar values are always identical.
    if kl_raw is not None and kl_raw < 0.1:
        warnings.append(f"KL raw very low ({kl_raw:.4f}) — posterior ≈ prior, latent may lack diversity")
    print(f"    continue_loss       : {_fmt(cont)}")
    print(f"    overshooting_kl     : {_fmt(overshoot_kl)}")
    print(f"    total_loss          : {_fmt(total)}")
    print(f"    grad_norm           : {_fmt(wm_gn)}")

    if wm_gn is not None and wm_gn < 1e-5:
        warnings.append("WM grad_norm near zero — world model may have stopped learning")
    if wm_gn is not None and wm_gn > 90.0:
        warnings.append(f"WM grad_norm very high ({wm_gn:.1f}) — approaching clip limit, training may be unstable")
    if recon is not None and recon > 0.1:
        warnings.append(f"Reconstruction loss still high ({recon:.4f}) — encoder/decoder underperforming")

    # ── Actor (behavior) ─────────────────────────────────────────────
    print(f"\n  {sep}")
    print("  ACTOR")
    print(f"  {sep}")
    act_loss = beh.get("actor_loss")
    act_gn = beh.get("actor_grad_norm")
    imag_rew_mean = beh.get("imagined_reward_mean")
    imag_rew_min = beh.get("imagined_reward_min")
    imag_rew_max = beh.get("imagined_reward_max")
    imag_rew_std = beh.get("imagined_reward_std")
    imag_val_mean = beh.get("imagined_value_mean")
    imag_val_std = beh.get("imagined_value_std")
    imag_ret_mean = beh.get("imagined_return_mean")
    imag_ret_std = beh.get("imagined_return_std")
    imag_act_mean = beh.get("imagined_action_mean")
    imag_act_std = beh.get("imagined_action_std")

    print(f"    actor_loss          : {_fmt(act_loss)}")
    print(f"    grad_norm           : {_fmt(act_gn)}")
    print(f"    imagined_reward     : mean={_fmt(imag_rew_mean)}  min={_fmt(imag_rew_min)}  max={_fmt(imag_rew_max)}  std={_fmt(imag_rew_std)}")
    print(f"    imagined_values     : mean={_fmt(imag_val_mean)}  std={_fmt(imag_val_std)}")
    print(f"    imagined_returns    : mean={_fmt(imag_ret_mean)}  std={_fmt(imag_ret_std)}")
    print(f"    imagined_actions    : mean={_fmt(imag_act_mean)}  std={_fmt(imag_act_std)}")

    if act_gn is not None and act_gn < 1e-6:
        warnings.append("Actor grad_norm near zero — actor is NOT learning (no useful gradient signal)")
    if imag_rew_max is not None and imag_rew_max <= 0.0:
        warnings.append("All imagined rewards ≤ 0 — actor sees no positive outcomes to learn from")
    if imag_rew_std is not None and imag_rew_std < 1e-4:
        warnings.append("Imagined reward std ≈ 0 — reward predictor may be collapsed to a constant")
    if imag_act_std is not None and imag_act_std < 0.01:
        warnings.append(f"Action std very low ({imag_act_std:.4f}) — actor may have collapsed (no exploration)")
    if imag_ret_std is not None and imag_ret_std < 1e-4:
        warnings.append("Return std ≈ 0 — all imagined trajectories look the same to the actor")

    if beh.get("actor_entropy") is not None:
        print(f"    actor_entropy       : {_fmt(beh['actor_entropy'])}")

    # ── Critic ───────────────────────────────────────────────────────
    print(f"\n  {sep}")
    print("  CRITIC")
    print(f"  {sep}")
    crit_loss = beh.get("critic_loss")
    crit_gn = beh.get("critic_grad_norm")
    print(f"    critic_loss         : {_fmt(crit_loss)}")
    print(f"    grad_norm           : {_fmt(crit_gn)}")

    if crit_gn is not None and crit_gn < 1e-6:
        warnings.append("Critic grad_norm near zero — critic is not fitting value targets")
    if crit_loss is not None and crit_loss > 50.0:
        warnings.append(f"Critic loss very high ({crit_loss:.2f}) — value estimates may be wildly inaccurate")

    # ── Evaluation ───────────────────────────────────────────────────
    if ev:
        print(f"\n  {sep}")
        print("  EVALUATION")
        print(f"  {sep}")
        print(f"    mean_reward         : {_fmt(ev.get('mean_reward'))}")
        print(f"    mean_steps          : {_fmt(ev.get('mean_steps'))}")
        print(f"    crash_rate          : {_fmt(ev.get('crash_rate'))}")
        crash = ev.get("crash_rate")
        mean_steps = ev.get("mean_steps")
        if crash is not None and crash > 0.8:
            warnings.append(f"Crash rate {crash:.0%} — agent crashes in most episodes")
        if mean_steps is not None and mean_steps < 20:
            warnings.append(f"Mean episode length only {mean_steps:.0f} steps — agent dies immediately")

    # ── Warning summary ──────────────────────────────────────────────
    if warnings:
        print(f"\n  {'⚠' * 3} WARNINGS {'⚠' * 3}")
        for i, w in enumerate(warnings, 1):
            print(f"    {i}. {w}")
    else:
        print(f"\n  ✓ No warnings — all levels look healthy")
    print(f"[DEBUG] {'═' * 62}\n", flush=True)


@dataclass(slots=True)
class TrainingRunSummary:
    """Summary returned after a training experiment completes.

    Contains final replay size, the latest checkpoint path, and the
    flattened metrics dict from the last cycle.
    """

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
    actor: Actor | DiscreteActor,
    *,
    episodes: int,
    max_steps: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Run deterministic evaluation episodes and aggregate statistics.

    Sets both the world model and actor to eval mode, runs ``episodes``
    rollouts with no gradient tracking, and returns mean reward, mean
    steps, and crash rate.  The original training mode is restored
    afterwards.
    """
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
        with torch.inference_mode():
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
    """Map a config device string to a torch.device, falling back to CPU."""
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def infer_env_shapes(config: ExperimentConfig) -> tuple[tuple[int, int, int], int]:
    """Probe a temporary environment to discover observation and action shapes.

    Creates the environment, resets it, and inspects:
    - ``observation.shape`` → ``(C, H, W)`` for the CNN encoder.
    - ``action_space.n`` (discrete) or ``action.shape[0]`` (continuous)
      → ``action_dim`` for the actor and RSSM.

    The environment is closed before returning.
    """
    env = make_highway_env(config.env)
    try:
        observation, _ = env.reset(seed=config.seed)
        if config.env.action.is_discrete:
            import gymnasium as gym
            assert isinstance(env.action_space, gym.spaces.Discrete)
            action_dim = int(env.action_space.n)
        else:
            action = env.action_space.sample()
            action_dim = int(action.shape[0])
    finally:
        env.close()

    observation_shape = tuple(int(dim) for dim in observation.shape)
    return observation_shape, action_dim


def _make_optimizer(
    params,
    lr: float,
) -> torch.optim.Optimizer:
    """Create a standard AdamW optimizer."""
    return torch.optim.AdamW(list(params), lr=lr)


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
    Actor | DiscreteActor,
    Critic,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    """Instantiate all training components from a config.

    Creates the replay buffer, world model, actor (continuous or
    discrete depending on ``config.env.action.is_discrete``), critic,
    and three AdamW optimizers.  All models are moved to the configured
    device.  When CUDA is available, float32 matmul precision is set
    to ``"high"`` for speed.

    Returns:
        7-tuple of (replay_buffer, world_model, actor, critic,
        world_model_optimizer, actor_optimizer, critic_optimizer).
    """
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
        num_categoricals=mc.num_categoricals,
        num_classes=mc.num_classes,
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
    if config.env.action.is_discrete:
        actor: Actor | DiscreteActor = DiscreteActor(
            latent_dim=latent_dim,
            num_actions=action_dim,
            hidden_dim=mc.actor_hidden_dim,
            num_layers=mc.actor_num_layers,
        ).to(device)
    else:
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

    world_model_optimizer = _make_optimizer(
        world_model.parameters(),
        lr=config.training.world_model_lr,
    )
    actor_optimizer = _make_optimizer(actor.parameters(), lr=config.training.actor_lr)
    critic_optimizer = _make_optimizer(critic.parameters(), lr=config.training.critic_lr)
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
    """Run a complete DreamerV1 training experiment.

    This is the highest-level training API.  It handles:

    * Global seed initialization for reproducibility.
    * Model and optimizer creation (or resumption from checkpoint).
    * LR warm-up scheduler setup and state restoration.
    * AMP (automatic mixed precision) configuration.
    * The main training loop with periodic evaluation, checkpointing,
      and progress logging.
    * Cleanup of stale artifacts when starting fresh.

    Args:
        config:              Full experiment configuration.
        artifact_root:       Directory for checkpoints, logs, and outputs.
        cycles:              Override ``config.training.cycles``.
        warm_start_steps:    Override ``config.training.warm_start_steps``.
        policy_steps:        Override ``config.training.policy_steps``.
        checkpoint_interval: Override ``config.training.checkpoint_interval``.
        resume_from:         Path to a checkpoint file to resume from.
        show_progress:       Whether to print per-cycle status lines.

    Returns:
        ``TrainingRunSummary`` with final statistics and paths.
    """
    set_global_seeds(config.seed, deterministic_torch=config.training.deterministic_torch)

    total_cycles = config.training.cycles if cycles is None else cycles
    initial_warm_start_steps = config.training.warm_start_steps if warm_start_steps is None else warm_start_steps
    cycle_policy_steps = config.training.policy_steps if policy_steps is None else policy_steps
    save_every = config.training.checkpoint_interval if checkpoint_interval is None else checkpoint_interval

    artifact_directory = Path(artifact_root)

    # Resolve the resume path early so we can check whether the file
    # actually exists.  A missing checkpoint is treated as a fresh run:
    # old artifacts are wiped and training starts from step 1.
    resolved_resume: Path | None = None
    if resume_from is not None:
        candidate = Path(resume_from)
        if candidate.exists():
            resolved_resume = candidate
        else:
            print(
                f"[train] checkpoint not found at '{candidate}' — "
                "starting a fresh run and clearing old artifacts.",
                flush=True,
            )

    # Clear stale artifacts only for truly fresh runs
    if resolved_resume is None and artifact_directory.exists():
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
    if resolved_resume is not None:
        metadata = load_checkpoint(
            resolved_resume,
            world_model=world_model,
            actor=actor,
            critic=critic,
            world_model_optimizer=world_model_optimizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            map_location=resolve_training_device(config.device),
            replay_buffer=replay_buffer,
        )
        start_step = int(metadata["step"]) + 1

    # LR warm-up schedulers: linearly ramp learning rate from ~0 to the
    # base LR over the configured number of optimizer steps.  Returns
    # None when warmup_steps == 0 (no warmup).
    wm_scheduler = _make_warmup_scheduler(world_model_optimizer, config.training.lr_warmup_steps)
    actor_scheduler = _make_warmup_scheduler(actor_optimizer, config.training.lr_warmup_steps)
    critic_scheduler = _make_warmup_scheduler(critic_optimizer, config.training.lr_warmup_steps)

    # Restore scheduler state from checkpoint so warmup progression
    # continues correctly from where it left off.
    if resolved_resume is not None:
        saved_schedulers = metadata.get("schedulers")
        if saved_schedulers is not None:
            if wm_scheduler is not None and "wm_scheduler" in saved_schedulers:
                wm_scheduler.load_state_dict(saved_schedulers["wm_scheduler"])
            if actor_scheduler is not None and "actor_scheduler" in saved_schedulers:
                actor_scheduler.load_state_dict(saved_schedulers["actor_scheduler"])
            if critic_scheduler is not None and "critic_scheduler" in saved_schedulers:
                critic_scheduler.load_state_dict(saved_schedulers["critic_scheduler"])

    # AMP — automatic mixed precision: run forward/backward in bfloat16
    # or float16 for faster compute and lower memory on CUDA GPUs.
    # bfloat16 does not require gradient scaling; float16 does.
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

        # Step LR warm-up schedulers once per *optimizer step*, not once
        # per cycle.  Each cycle runs N world-model steps and M behavior
        # steps, so we advance the schedulers by those counts.
        n_wm = config.training.world_model_updates_per_cycle
        n_beh = config.training.behavior_updates_per_cycle
        if wm_scheduler is not None:
            for _ in range(n_wm):
                wm_scheduler.step()
        if actor_scheduler is not None:
            for _ in range(n_beh):
                actor_scheduler.step()
        if critic_scheduler is not None:
            for _ in range(n_beh):
                critic_scheduler.step()

        checkpoint_file = None
        if step % save_every == 0 or step == total_cycles:
            flattened = flatten_cycle_metrics(step, latest_metrics)
            scheduler_states: dict[str, Any] = {}
            if wm_scheduler is not None:
                scheduler_states["wm_scheduler"] = wm_scheduler.state_dict()
            if actor_scheduler is not None:
                scheduler_states["actor_scheduler"] = actor_scheduler.state_dict()
            if critic_scheduler is not None:
                scheduler_states["critic_scheduler"] = critic_scheduler.state_dict()
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
                replay_buffer=replay_buffer,
                schedulers=scheduler_states if scheduler_states else None,
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

            if (
                config.training.debug_logging == "verbose"
                and step % config.training.debug_print_every == 0
            ):
                _print_debug_cycle_details(
                    step,
                    total_cycles,
                    latest_metrics,
                    world_model_optimizer=world_model_optimizer,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    checkpoint_file=checkpoint_file,
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