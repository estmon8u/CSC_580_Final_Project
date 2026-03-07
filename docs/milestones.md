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

**Status:** Completed.

Completed work:

- `ObservationEncoder` (CNN → 1024-dim embedding)
- `RecurrentStateSpaceModel` with multi-layer prior/posterior MLPs (default 2 layers)
- decoder and reward head (configurable layers/hidden dims)
- combined `TinyWorldModel` with Kaiming uniform weight initialization
- `ModelConfig` for all configurable model dimensions (DreamerV1-reference defaults)
- single-step world-model training helper
- short sequence world-model training helper
- KL divergence term with free-nats handling
- notebook smoke tests up through short sequence training

## M4

Implement imagination rollouts, actor, critic, and TD-lambda targets.

**Status:** Completed.

Completed work:

- latent imagination rollout helper
- `Actor` module with TanhTransform distribution (`init_std=5.0`, `mean_scale=5.0`, `min_std=1e-4`)
- `Critic` module with configurable depth (default 3 hidden layers)
- `td_lambda_returns()` with corrected next-state value formula
- behavior training helper
- behavior update with gradient isolation from the world model
- WM posterior passthrough to behavior learning (avoids redundant re-encoding)
- deterministic behavior smoke tests
- notebook smoke test for imagined trajectories

## M5

Run the alternating train-collect pipeline and export evaluation media.

**Status:** Completed.

Completed work:

- minimal alternating training cycle helper
- actor-driven environment collection helper
- pipeline smoke test covering warm start, one world-model update, one behavior update, and policy collection
- notebook smoke test for the alternating pipeline step
- checkpoint save/load helpers
- checkpoint smoke tests
- checkpoint notebook smoke test
- lightweight metrics export helpers
- JSONL, CSV, and latest-summary artifact writers
- metrics logging tests
- notebook metrics export smoke test
- n-step prediction evaluation helpers
- prediction evaluation tests
- notebook n-step prediction evaluation smoke test
- metric plot artifact helper
- predicted-vs-target comparison grid helper
- predicted-vs-target comparison video helper
- visualization tests
- notebook plot artifact smoke test
- notebook video artifact smoke test
- submission bundle helper
- artifact bundle tests
- notebook submission bundle smoke test

## M6

Run real training experiments, tune the baseline, and analyze learning behavior.

**Status:** In progress.

Completed work:

- baseline training runner with checkpoint/log export
- DreamerV1 reference implementation comparison and staged integration plan
- critical TD(λ) return formula fix (was using current-state values, now uses next-state)
- actor exploration fix (init_std=5.0, mean_scale=5.0, TanhTransform with Jacobian)
- model capacity aligned to DreamerV1 reference (embedding 1024, det 200, stoch 30)
- multi-layer networks: RSSM 2-layer, critic 3-layer, reward predictor 2-layer
- Kaiming uniform weight initialization across all networks
- WM posterior passthrough to behavior learning
- ModelConfig with all configurable dimensions wired through experiment and evaluation
- sequence length increased to 32, imagination horizon to 15
- learned continuation model and continuation-aware returns
- probabilistic reward and critic heads with likelihood losses
- probabilistic observation decoder with observation-NLL metrics
- latent overshooting / multi-step consistency regularization
- latent rollout consistency evaluation for longer-horizon drift checks
- AMP (bfloat16) and FlashAdamW support for H100
- 7 Colab notebooks covering smoke tests through H100 screening
- 6 YAML config profiles (CPU, production, optimized, H100, H100+AMP, screening)
- full test suite passing

Still in progress:

- H100 screening run with reference-aligned code
- inspection of reward trends and world-model losses
- comparison of early and late checkpoints
- determination of whether the agent shows learning progress

## Phase checkpoint summary

Completed coding phases recorded in git:

- Phase 1 — continuation model, continuation-aware returns, replay fixes, evaluation integration
- Phase 2 — probabilistic reward and value heads
- Phase 3 — probabilistic observation modeling
- Phase 4 — latent overshooting consistency

From this point forward, additional phases are optional. The project can shift from implementation phases to experiment phases and final-deliverable phases.

## M7

Prepare final deliverables after training results are credible.

**Status:** Not started.

Planned work:

- curate final plots, tables, images, and videos
- clean the notebook for presentation quality
- assemble the final submission bundle and Drive folder
- write the final report and reflection

## Optimization policy

Most engineering and performance optimizations that were previously deferred have been implemented:

- ✅ AMP / autocast (bfloat16 on H100, fp16 with GradScaler on older GPUs)
- ✅ FlashAdamW fused optimizer
- ✅ Kaiming weight initialization
- ✅ Multi-layer networks matching DreamerV1 reference capacity
- ✅ Configurable model dimensions via `ModelConfig`

Still deferred:

- WandB logging
- Hydra config management
- `torch.compile`
- prioritized replay
