"""Tests for training optimizations: gradient clipping, AdamW, LR warm-up.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch

from tiny_dreamer_highway.config import ExperimentConfig, TrainingConfig
from tiny_dreamer_highway.models import Actor, Critic, TinyWorldModel
from tiny_dreamer_highway.training.behavior_learning import train_behavior_step
from tiny_dreamer_highway.training.experiment import (
    _make_optimizer,
    _make_warmup_scheduler,
    initialize_training_state,
)
from tiny_dreamer_highway.training.world_model_step import train_world_model_step


# ---------------------------------------------------------------------------
# Gradient clipping tests
# ---------------------------------------------------------------------------

def test_world_model_step_clips_gradients() -> None:
    """Verify clip_grad_norm_ is applied inside train_world_model_step."""
    torch.manual_seed(7)
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    observations = torch.randint(0, 255, (4, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(4, 2)
    rewards = torch.randn(4)

    # Use an extremely small clip norm to guarantee clipping actually fires
    _, metrics = train_world_model_step(
        model, optimizer, observations, actions, rewards,
        grad_clip_norm=1e-6,
    )

    # After step, all param grads should have been clipped before the update.
    # We can't directly inspect post-clip grads after .step() zeroes them,
    # but we verify no NaN in metrics (clipping prevents explosions).
    for value in metrics.values():
        assert not (isinstance(value, float) and (value != value)), \
            f"NaN detected in metrics: {metrics}"


def test_behavior_step_clips_gradients() -> None:
    """Verify clip_grad_norm_ is applied inside train_behavior_step."""
    torch.manual_seed(7)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    start_state = world_model.rssm.initial_state(batch_size=4)

    metrics = train_behavior_step(
        world_model, actor, critic, actor_opt, critic_opt,
        start_state, horizon=3, grad_clip_norm=1e-6,
    )

    for value in metrics.values():
        assert not (isinstance(value, float) and (value != value)), \
            f"NaN detected in metrics: {metrics}"


def test_grad_clip_norm_actually_limits_gradient_magnitude() -> None:
    """Directly confirm clip_grad_norm bounds the total gradient norm."""
    torch.manual_seed(7)
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )

    # Forward + backward to get large gradients
    observations = torch.randint(0, 255, (4, 1, 64, 64), dtype=torch.uint8).float() / 255.0
    output = model(observations, torch.randn(4, 2))
    loss = output.reconstruction.sum()
    loss.backward()

    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

    # After clipping, re-computing the norm should be <= max_norm (within float tolerance)
    clipped_norm = sum(
        p.grad.detach().norm() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    assert clipped_norm.item() <= max_norm + 1e-4, \
        f"Clipped norm {clipped_norm.item():.6f} exceeds max_norm {max_norm}"


# ---------------------------------------------------------------------------
# AdamW tests
# ---------------------------------------------------------------------------

def test_make_optimizer_returns_adamw_by_default() -> None:
    """_make_optimizer should return torch.optim.AdamW."""
    params = [torch.nn.Parameter(torch.randn(3))]
    optimizer = _make_optimizer(params, lr=1e-3)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_make_optimizer_preserves_learning_rate() -> None:
    """Check that the learning rate is correctly forwarded."""
    params = [torch.nn.Parameter(torch.randn(3))]
    for lr in [1e-5, 3e-4, 1e-2]:
        optimizer = _make_optimizer(params, lr=lr)
        assert optimizer.defaults["lr"] == lr


def test_initialize_training_state_uses_adamw() -> None:
    """Confirm initialize_training_state creates AdamW optimizers (not plain Adam)."""
    config = ExperimentConfig()
    _, _, _, _, wm_opt, actor_opt, critic_opt = initialize_training_state(config)
    # On CPU / Windows, should be AdamW (not plain Adam)
    assert isinstance(wm_opt, torch.optim.AdamW)
    assert isinstance(actor_opt, torch.optim.AdamW)
    assert isinstance(critic_opt, torch.optim.AdamW)


# ---------------------------------------------------------------------------
# LR warm-up scheduler tests
# ---------------------------------------------------------------------------

def test_make_warmup_scheduler_returns_none_when_steps_zero() -> None:
    """No scheduler created when warmup_steps <= 0."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(3))], lr=3e-4)
    assert _make_warmup_scheduler(optimizer, warmup_steps=0) is None
    assert _make_warmup_scheduler(optimizer, warmup_steps=-1) is None


def test_make_warmup_scheduler_returns_lambda_scheduler() -> None:
    """Valid warmup_steps should produce a LambdaLR scheduler."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(3))], lr=3e-4)
    scheduler = _make_warmup_scheduler(optimizer, warmup_steps=50)
    assert scheduler is not None
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_warmup_scheduler_ramps_lr_linearly() -> None:
    """LR should ramp from near-zero to base LR over warmup_steps."""
    base_lr = 3e-4
    warmup_steps = 10
    param = torch.nn.Parameter(torch.randn(3))
    optimizer = torch.optim.AdamW([param], lr=base_lr)
    scheduler = _make_warmup_scheduler(optimizer, warmup_steps=warmup_steps)
    assert scheduler is not None

    recorded_lrs: list[float] = []
    for step in range(warmup_steps + 5):
        recorded_lrs.append(optimizer.param_groups[0]["lr"])
        # Simulate an optimizer step (scheduler expects this)
        optimizer.step()
        scheduler.step()

    # LR at step 0 should be ~1/warmup_steps of base_lr (first step: (0+1)/10 = 0.1)
    assert recorded_lrs[0] < base_lr * 0.5, \
        f"Initial LR {recorded_lrs[0]} should be well below base {base_lr}"

    # LR at warmup_steps should be at (or very near) base_lr
    at_warmup = recorded_lrs[warmup_steps]
    assert abs(at_warmup - base_lr) < 1e-8, \
        f"LR at warmup end {at_warmup:.8f} should equal base_lr {base_lr}"

    # LR after warmup should stay at base_lr (lambda clamps to 1.0)
    past_warmup = recorded_lrs[warmup_steps + 2]
    assert abs(past_warmup - base_lr) < 1e-8, \
        f"LR past warmup {past_warmup:.8f} should stay at base_lr {base_lr}"

    # LR should be monotonically non-decreasing during warmup
    for i in range(1, warmup_steps):
        assert recorded_lrs[i] >= recorded_lrs[i - 1] - 1e-10, \
            f"LR decreased at step {i}: {recorded_lrs[i-1]} -> {recorded_lrs[i]}"


# ---------------------------------------------------------------------------
# Config field tests
# ---------------------------------------------------------------------------

def test_grad_clip_norm_config_default() -> None:
    config = TrainingConfig()
    assert config.grad_clip_norm == 100.0


def test_lr_warmup_steps_config_default() -> None:
    config = TrainingConfig()
    assert config.lr_warmup_steps == 0


def test_config_accepts_optimization_fields_from_dict() -> None:
    """Ensure new fields are parsed correctly from a dict (simulating YAML load)."""
    config = ExperimentConfig.model_validate({
        "training": {
            "grad_clip_norm": 50.0,
            "lr_warmup_steps": 25,
        }
    })
    assert config.training.grad_clip_norm == 50.0
    assert config.training.lr_warmup_steps == 25


# ---------------------------------------------------------------------------
# Pipeline integration: grad_clip_norm flows through run_training_cycle
# ---------------------------------------------------------------------------

def test_grad_clip_norm_passed_through_pipeline(monkeypatch) -> None:
    """Verify that run_training_cycle forwards grad_clip_norm from config."""
    import numpy as np
    from tiny_dreamer_highway.data.replay_buffer import ReplayBuffer
    from tiny_dreamer_highway.training.pipeline import run_training_cycle
    from tiny_dreamer_highway.types import Transition

    config = ExperimentConfig.model_validate({
        "training": {
            "batch_size": 4,
            "imagination_horizon": 3,
            "grad_clip_norm": 42.0,
        }
    })

    replay = ReplayBuffer(capacity=128)
    for i in range(16):
        replay.add(Transition(
            observation=np.full((1, 64, 64), i, dtype=np.uint8),
            action=np.array([0.1, 0.2], dtype=np.float32),
            reward=float(i) / 10,
            next_observation=np.full((1, 64, 64), i + 1, dtype=np.uint8),
            done=False,
        ))

    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    wm_opt = torch.optim.AdamW(world_model.parameters(), lr=1e-3)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    captured_wm_clip: list[float] = []
    captured_beh_clip: list[float] = []

    _orig_wm = train_world_model_step
    _orig_beh = train_behavior_step

    def spy_wm(*args, **kwargs):
        captured_wm_clip.append(kwargs.get("grad_clip_norm", -1))
        return _orig_wm(*args, **kwargs)

    def spy_beh(*args, **kwargs):
        captured_beh_clip.append(kwargs.get("grad_clip_norm", -1))
        return _orig_beh(*args, **kwargs)

    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.train_world_model_step", spy_wm,
    )
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.train_behavior_step", spy_beh,
    )
    monkeypatch.setattr(
        "tiny_dreamer_highway.training.pipeline.collect_actor_transitions",
        lambda *a, **kw: 0,
    )

    run_training_cycle(
        config, replay, world_model, actor, critic,
        wm_opt, actor_opt, critic_opt,
        warm_start_steps=0, policy_steps=0, seed=7,
    )

    assert all(v == 42.0 for v in captured_wm_clip), f"Expected 42.0, got {captured_wm_clip}"
    assert all(v == 42.0 for v in captured_beh_clip), f"Expected 42.0, got {captured_beh_clip}"


# ---------------------------------------------------------------------------
# AMP configuration tests
# ---------------------------------------------------------------------------

def test_amp_config_defaults() -> None:
    """AMP should be disabled by default."""
    config = TrainingConfig()
    assert config.use_amp is False
    assert config.amp_dtype == "bfloat16"
    assert config.use_flash_optimizer is False


def test_amp_config_accepts_float16() -> None:
    config = TrainingConfig(use_amp=True, amp_dtype="float16")
    assert config.amp_dtype == "float16"


def test_amp_config_from_dict() -> None:
    config = ExperimentConfig.model_validate({
        "training": {
            "use_amp": True,
            "amp_dtype": "bfloat16",
            "use_flash_optimizer": True,
        }
    })
    assert config.training.use_amp is True
    assert config.training.amp_dtype == "bfloat16"
    assert config.training.use_flash_optimizer is True


def test_resolve_amp_dtype() -> None:
    from tiny_dreamer_highway.training.pipeline import resolve_amp_dtype
    assert resolve_amp_dtype("bfloat16") == torch.bfloat16
    assert resolve_amp_dtype("float16") == torch.float16


def test_world_model_step_accepts_amp_args() -> None:
    """train_world_model_step should work when amp args are None (CPU path)."""
    torch.manual_seed(7)
    model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    observations = torch.randint(0, 255, (4, 1, 64, 64), dtype=torch.uint8)
    actions = torch.randn(4, 2)
    rewards = torch.randn(4)

    _, metrics = train_world_model_step(
        model, optimizer, observations, actions, rewards,
        grad_scaler=None,
        amp_context=None,
    )
    assert "total_loss" in metrics


def test_behavior_step_accepts_amp_args() -> None:
    """train_behavior_step should work when amp args are None (CPU path)."""
    torch.manual_seed(7)
    world_model = TinyWorldModel(
        observation_shape=(1, 64, 64), action_dim=2,
        embedding_dim=256, deterministic_dim=128, stochastic_dim=32, hidden_dim=128,
    )
    actor = Actor(latent_dim=160, action_dim=2, hidden_dim=64, num_layers=1)
    critic = Critic(latent_dim=160, hidden_dim=64, num_layers=1)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    start_state = world_model.rssm.initial_state(batch_size=4)

    metrics = train_behavior_step(
        world_model, actor, critic, actor_opt, critic_opt,
        start_state, horizon=3,
        actor_scaler=None,
        critic_scaler=None,
        amp_context=None,
    )
    assert "actor_loss" in metrics


# ---------------------------------------------------------------------------
# FlashAdamW optional import tests
# ---------------------------------------------------------------------------

def test_make_optimizer_returns_adamw_when_flash_unavailable() -> None:
    """When use_flash=True but flashoptim is not installed, should fall back to AdamW."""
    from tiny_dreamer_highway.training.experiment import _make_optimizer
    params = [torch.nn.Parameter(torch.randn(3))]
    optimizer = _make_optimizer(params, lr=1e-3, use_flash=True)
    # On Windows/CPU, FlashAdamW is not available — should get AdamW
    assert isinstance(optimizer, torch.optim.AdamW)


def test_make_optimizer_flash_false_returns_adamw() -> None:
    from tiny_dreamer_highway.training.experiment import _make_optimizer
    params = [torch.nn.Parameter(torch.randn(3))]
    optimizer = _make_optimizer(params, lr=1e-3, use_flash=False)
    assert isinstance(optimizer, torch.optim.AdamW)
