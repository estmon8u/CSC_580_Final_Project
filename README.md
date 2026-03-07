# Tiny Dreamer Highway

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

A test-first, package-based implementation scaffold for a Dreamer V1-style world model trained on `highway-v0` from Highway-Env.

## Goals

- structure the final project like a professional Python project
- build the system incrementally and validate each step before integration
- keep notebooks for presentation and exploration, while reusable code lives in `src/`
- generate report-ready metrics, plots, and n-step prediction videos

## Project structure

- `docs/` architecture notes, milestones, risks, and evaluation checkpoints
- `examples/` experiment configuration files (CPU smoke, production, H100, screening)
- `notebooks/` polished exploratory and presentation notebooks (01–07)
- `src/tiny_dreamer_highway/` reusable package code
- `tests/` fast tests for config, replay, models, pipeline, and behavior
- `artifacts/` checkpoints, plots, videos, and logs

## Current status

The codebase is a fully tested DreamerV1-style world model for `highway-v0`.

All core components are complete:
- typed experiment configuration with `ModelConfig` for tunable model dimensions
- replay buffer, environment factory, and random rollout collection
- CNN encoder, RSSM (multi-layer prior/posterior), decoder, reward predictor
- combined world model with Kaiming initialization
- probabilistic observation, reward, and value heads with fixed-std likelihood losses
- actor with TanhTransform, `init_std=5.0`, and `mean_scale=5.0` for exploration
- critic with configurable depth (default 3 layers)
- corrected TD(λ) returns using next-state values
- imagination rollouts with WM posterior passthrough to behavior learning
- alternating collect/train pipeline
- checkpoint save/load and metrics export
- n-step prediction evaluation, plots, videos, submission bundles, and observation-NLL reporting
- AMP (bfloat16) and FlashAdamW support for H100
- 113 passing tests

Model defaults match the DreamerV1 reference: `embedding_dim=1024`, `deterministic_dim=200`, `stochastic_dim=30`.

## Colab workflow

Use GitHub for source code and Google Drive for generated artifacts.

Recommended loop:

1. make changes locally in VS Code
2. run tests before every push
3. push validated changes to the personal GitHub repository
4. in Colab, mount Drive, clone or pull the repository, and install the package from `pyproject.toml` with `python -m pip install -e .`
5. write checkpoints, videos, and plots to the mounted Drive folder

The notebook should stay thin and call reusable functions from `src/tiny_dreamer_highway/`.

## Notebook set

- `01_colab_setup_and_smoke_tests.ipynb` — environment setup and smoke validation
- `02_colab_sanity_run.ipynb` — short runner validation job
- `03_colab_baseline_run.ipynb` — first real baseline comparison run
- `04_colab_h100_run.ipynb` — larger H100-oriented run
- `05_colab_optimized_run.ipynb` — optimized comparison (AdamW, grad clipping, LR warm-up)
- `06_colab_h100_amp_run.ipynb` — H100 + AMP (bfloat16) throughput run
- `07_colab_screening_run.ipynb` — safer H100 screening with DreamerV1-reference model

See [docs/architecture_overview.md](docs/architecture_overview.md) and [docs/milestones.md](docs/milestones.md) before extending the package.
