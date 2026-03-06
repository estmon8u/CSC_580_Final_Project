# Milestones

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

## M1

Scaffold the package, docs, config system, replay buffer, and tests.

**Status:** Completed.

Completed work:

- project scaffold under `src/`, `tests/`, `docs/`, `examples/`, and `notebooks/`
- `pyproject.toml` packaging and dependency management
- typed config models and initial CLI support
- local `pytest` workflow before push
- GitHub as code source of truth and Drive-backed Colab artifact workflow

## M2

Collect random Highway-Env rollouts and verify replay sampling.

**Status:** Completed.

Completed work:

- `highway_factory.py`
- replay buffer add and sample helpers
- random rollout collection
- seeded environment reset and action-space seeding
- replay and rollout tests

## M3

Implement the world model: encoder, RSSM, reward model, and decoder.

**Status:** Completed at baseline level.

Completed work:

- `ObservationEncoder`
- `RecurrentStateSpaceModel`
- decoder and reward head
- combined `TinyWorldModel`
- single-step world-model training helper
- short sequence world-model training helper
- notebook smoke tests up through short sequence training

Still missing inside M3:

- Dreamer-style KL divergence term
- free-nats handling
- richer world-model diagnostics and checkpointing

## M4

Implement imagination rollouts, actor, critic, and TD-lambda targets.

**Status:** In progress.

Completed baseline work:

- latent imagination rollout helper
- `Actor` module
- `Critic` module
- `td_lambda_returns()`
- behavior training helper
- deterministic behavior smoke tests
- notebook smoke test for imagined trajectories

Still missing:

- behavior loss helpers with gradient isolation from the world model
- richer Dreamer-style actor objective details and longer imagined training validation

## M5

Run the alternating train-collect pipeline and export evaluation media.

**Status:** In progress.

Planned work:

- alternating collect → world model train → behavior train → act loop
- checkpoint save/load
- metrics export
- n-step prediction evaluation
- plots, videos, and report-ready artifacts

Completed baseline work:

- minimal alternating training cycle helper
- actor-driven environment collection helper
- pipeline smoke test covering warm start, one world-model update, one behavior update, and policy collection
- checkpoint save/load helpers
- checkpoint smoke tests

## Optimization policy

Optional engineering and performance optimizations are intentionally deferred until the baseline pipeline works end to end.

Deferred items include:

- WandB logging
- Hydra config management
- `torch.compile`
- AMP / autocast
- `FlashAdamW` / `flashoptim`
- prioritized replay
- extra performance tuning
