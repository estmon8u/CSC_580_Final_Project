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

- Completed baseline layers:
  - config, replay, env factory, tests
  - random rollout collection
  - encoder, RSSM, decoder, reward head
  - combined world-model forward pass
  - single-step and short-sequence world-model training helpers
  - reproducibility seeding
  - combined Colab setup-and-smoke-test notebook

- Next core layers:
  - imagination rollouts in latent space
  - actor and critic modules
  - TD-lambda targets and behavior losses
  - alternating train/collect pipeline
  - evaluation metrics, plots, and videos

- Deferred until after baseline completion:
  - WandB
  - Hydra
  - `torch.compile`
  - AMP / autocast
  - optimizer and replay performance upgrades

## Notebook role

The notebook layer is intentionally thin.

Current notebook usage focuses on:

- environment and dependency sanity checks
- replay warm-start validation
- world-model smoke tests
- Colab execution against pushed package code

## Design rules

- keep notebooks for exploration, not for core source code
- keep tests small and deterministic
- validate contracts and shapes before integration
- store experiment outputs in `artifacts/`
- finish the baseline end-to-end path before optional optimizations
