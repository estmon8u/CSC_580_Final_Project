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

- Completed baseline smoke-tested layers:
  - config, replay, env factory, tests
  - random rollout collection
  - encoder, RSSM, decoder, reward head
  - combined world-model forward pass
  - single-step and short-sequence world-model training helpers
  - imagination rollouts, actor, critic, and TD-lambda behavior learning
  - alternating collect/train pipeline
  - checkpoint save/load and metrics export
  - n-step prediction evaluation
  - plot, comparison-grid, and GIF artifact export
  - manifest-backed submission bundle export
  - reproducibility seeding
  - combined Colab setup-and-smoke-test notebook

- What is not done yet:
  - sustained end-to-end training runs
  - evidence that the policy actually improves beyond random behavior
  - longer-horizon evaluation on trained checkpoints
  - training-stability tuning and quality-focused optimization
  - final demo generation from a genuinely trained agent
  - final report polish and submission packaging

- Deferred until after baseline completion:
  - WandB
  - Hydra
  - `torch.compile`
  - AMP / autocast
  - optimizer and replay performance upgrades

## Phase interpretation

The current codebase should be treated as a validated baseline scaffold, not a finished trained project.

So far, the work has focused on proving that the pieces connect correctly:

- tensors have the expected shapes
- optimization steps run without errors
- checkpoints and metrics can be exported
- notebook smoke tests run against pushed code

The next real project phase is to run training campaigns, inspect learning behavior, and tune the system until the agent and world model show meaningful results.

The project now includes a baseline training runner for that phase: it can initialize the model stack, execute multi-cycle training, and write checkpoints plus metrics logs to an artifact directory.

Real training runs should be launched from a dedicated Colab notebook so the smoke-test notebook remains focused on lightweight validation only.

## Notebook role

The notebook layer is intentionally thin.

Current notebook usage focuses on:

- environment and dependency sanity checks
- replay warm-start validation
- world-model and behavior smoke tests
- pipeline, checkpoint, and metrics smoke tests
- prediction evaluation, plots, video, and submission-bundle smoke tests
- Colab execution against pushed package code

The current setup notebook is still a smoke-test notebook, not the training-run notebook.

Actual training runs should live in a separate Colab notebook so longer jobs, checkpoint generation, and experiment notes do not clutter the smoke-test path.

## Design rules

- keep notebooks for exploration, not for core source code
- keep tests small and deterministic
- validate contracts and shapes before integration
- store experiment outputs in `artifacts/`
- finish the baseline end-to-end path before optional optimizations
