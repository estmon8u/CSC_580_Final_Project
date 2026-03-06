# Architecture Overview

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

## Phase 1: Test-first scaffold

The initial phase establishes a reusable Python package, experiment configuration, environment factory, and replay buffer before any heavy Dreamer training work.

### Main layers

1. `examples/` for reproducible configuration
2. `src/tiny_dreamer_highway/config.py` for typed settings
3. `src/tiny_dreamer_highway/envs/` for Highway-Env creation and preprocessing hooks
4. `src/tiny_dreamer_highway/data/` for replay and rollout storage
5. `tests/` for fast validation

## Intended system growth

- Phase 1: config, replay, env factory, tests
- Phase 2: encoder, decoder, RSSM, reward model
- Phase 3: imagination, actor, critic, TD-lambda
- Phase 4: training pipeline, evaluation, video exports

## Design rules

- keep notebooks for exploration, not for core source code
- keep tests small and deterministic
- validate contracts and shapes before integration
- store experiment outputs in `artifacts/`
