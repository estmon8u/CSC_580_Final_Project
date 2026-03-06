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
- `examples/` starter experiment configuration files
- `notebooks/` polished exploratory and presentation notebooks
- `src/tiny_dreamer_highway/` reusable package code
- `tests/` fast tests for config, replay, and smoke validation
- `artifacts/` checkpoints, plots, videos, and logs

## Initial milestone

The first implementation milestone focuses on:

1. project scaffolding
2. typed experiment configuration
3. replay buffer implementation
4. Highway-Env factory setup
5. fast unit tests

## Next steps

1. create the Python environment and install dependencies
2. run the test suite
3. implement random rollout collection
4. add the world model modules

## First real training loop

The project now includes a first baseline training runner so the next phase can move beyond smoke tests.

Per project workflow, actual training runs should be launched from a dedicated Google Colab notebook rather than from local ad hoc terminal commands.

The training runner initializes the replay buffer, world model, actor, and critic, then runs multi-cycle training while writing logs and checkpoints under the chosen artifact directory.

## Colab workflow

Use GitHub for source code and Google Drive for generated artifacts.

Recommended loop:

1. make changes locally in VS Code
2. run tests before every push
3. push validated changes to the personal GitHub repository
4. in Colab, mount Drive, clone or pull the repository, and install the package from `pyproject.toml` with `python -m pip install -e .`
5. write checkpoints, videos, and plots to the mounted Drive folder

The notebook should stay thin and call reusable functions from `src/tiny_dreamer_highway/`.

Recommended notebook split:

- `01_colab_setup_and_smoke_tests.ipynb` for environment setup and smoke validation
- `02_colab_training_runs.ipynb` for actual training experiments and checkpoint generation

See [docs/architecture_overview.md](docs/architecture_overview.md) and [docs/milestones.md](docs/milestones.md) before extending the package.
