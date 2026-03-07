# Architecture Overview

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

## Current project shape

The current implementation is a package-first, test-first Tiny Dreamer baseline for `highway-v0`.

The working development loop is:

1. implement locally in VS Code
2. run local `pytest`
3. commit and push validated changes to GitHub
4. pull the repo in Colab
5. install with `python -m pip install -e .`
6. run thin notebook smoke tests
7. store generated artifacts in Google Drive

GitHub is the source of truth for code. Google Drive is the source of truth for generated artifacts.

## Phase 1: Test-first scaffold

The initial phase established the reusable Python package, experiment configuration, environment factory, replay buffer, and test harness before heavier Dreamer training work.

### Main layers

1. `examples/` for reproducible configuration
2. `src/tiny_dreamer_highway/config.py` for typed settings
3. `src/tiny_dreamer_highway/envs/` for Highway-Env creation
4. `src/tiny_dreamer_highway/data/` for replay and rollout storage
5. `src/tiny_dreamer_highway/models/` for the world-model baseline
6. `src/tiny_dreamer_highway/training/` for optimization helpers
7. `tests/` for fast validation

## Current implementation status

- Completed and tested layers:
  - config, replay, env factory, tests
  - random rollout collection
  - encoder, RSSM (multi-layer prior/posterior), decoder, reward head
  - combined world-model forward pass with Kaiming initialization
  - single-step and short-sequence world-model training helpers
  - imagination rollouts, actor, critic, and TD-lambda behavior learning
  - actor with TanhTransform distribution, `init_std=5.0`, `mean_scale=5.0` for exploration
  - critic with configurable depth (default 3 hidden layers)
  - corrected TD(λ) returns using next-state values
  - WM posterior passthrough to behavior learning (eliminates redundant re-encoding)
  - `ModelConfig` for configurable model dimensions (DreamerV1 reference defaults)
  - alternating collect/train pipeline
  - checkpoint save/load and metrics export
  - n-step prediction evaluation
  - plot, comparison-grid, and GIF artifact export
  - manifest-backed submission bundle export
  - reproducibility seeding
  - AMP (bfloat16/fp16) with autocast and GradScaler
  - FlashAdamW fused optimizer support
  - combined Colab setup-and-smoke-test notebook
  - 7 Colab notebooks (smoke test, sanity, baseline, H100, optimized, AMP, screening)
  - 113 passing tests

- What is not done yet:
  - sustained end-to-end training runs with confirmed learning progress
  - evidence that the policy actually improves beyond random behavior
  - longer-horizon evaluation on trained checkpoints
  - final demo generation from a genuinely trained agent
  - final report polish and submission packaging

## Model architecture (DreamerV1-aligned)

The model dimensions match the open-source DreamerV1 reference implementation:

| Component | Key dimensions |
|---|---|
| CNN Encoder | 4 conv layers → embedding_dim=1024 |
| RSSM | deterministic=200, stochastic=30, 2-layer prior/posterior MLPs |
| CNN Decoder | MLP → 4 deconv layers from latent (230-dim) |
| Reward Predictor | 2 hidden layers × 200 units |
| Actor | 2 hidden layers × 200, TanhTransform with init_std=5.0, mean_scale=5.0 |
| Critic | 3 hidden layers × 200 units |

All linear/conv weights use Kaiming uniform initialization.

All dimensions are configurable via the `model:` section in YAML configs (parsed into `ModelConfig`).

## Phase interpretation

The current codebase is a tested DreamerV1-aligned implementation with all core components in place.

The model architecture, initialization, and training formulas have been aligned against an open-source DreamerV1 reference. Key fixes applied:

- TD(λ) returns corrected to use next-state values (was using current values — caused value hallucination)
- Actor exploration widened with `init_std=5.0`, `mean_scale=5.0`, and proper TanhTransform Jacobian correction
- Model capacity increased to reference scale (embedding 1024, det 200, stoch 30, multi-layer networks)
- WM posteriors passed directly to behavior learning instead of redundant re-encoding
- Kaiming uniform weight initialization across all networks

The next project phase is to run training campaigns on H100, inspect learning behavior, and tune the system until the agent and world model show meaningful improvement over random behavior.

## Notebook role

The notebook layer is intentionally thin.

Current notebook set:

- `01_colab_setup_and_smoke_tests.ipynb` — environment and dependency sanity checks, replay warm-start, world-model and behavior smoke tests, pipeline and checkpoint smoke tests, prediction evaluation and submission bundle
- `02_colab_sanity_run.ipynb` — short runner validation job
- `03_colab_baseline_run.ipynb` — first real baseline comparison run
- `04_colab_h100_run.ipynb` — larger H100-oriented run
- `05_colab_optimized_run.ipynb` — optimized comparison (AdamW, grad clipping, LR warm-up)
- `06_colab_h100_amp_run.ipynb` — H100 + AMP (bfloat16) throughput run
- `07_colab_screening_run.ipynb` — safer H100 screening with DreamerV1-reference model

Actual training runs live in separate notebooks from the smoke-test notebook so longer jobs, checkpoint generation, and experiment notes do not clutter the validation path.

## Design rules

- keep notebooks for exploration, not for core source code
- keep tests small and deterministic
- validate contracts and shapes before integration
- store experiment outputs in `artifacts/`
- finish the baseline end-to-end path before optional optimizations
