# Dream the Road: Model-Based Reinforcement Learning on Highway-v0

**Author:** Esteban Montelongo  
**Course:** CSC 580 — Artificial Intelligence II, DePaul University (Winter 2026)  
**AI tools consulted:** GitHub Copilot, ChatGPT

## Overview

This repository contains a from-scratch implementation of a DreamerV1-style model-based reinforcement learning agent that learns to drive on the `highway-v0` environment from raw image observations. The agent learns a latent world model (RSSM), then trains an actor-critic policy entirely inside imagined rollouts — no simulator needed during behavior learning.

The project was developed using a test-driven workflow: all code was written and tested locally in VS Code with `pytest`, pushed to GitHub, then pulled and trained on Google Colab (H100 GPU). Training artifacts (checkpoints, logs, videos) are saved to Google Drive. Notebooks execute the pipeline but never duplicate source code.

## Repository Structure

```txt
src/tiny_dreamer_highway/   # All reusable source code (models, training, evaluation)
tests/                       # pytest test suite
notebooks/
  configs/                   # YAML experiment configurations
  final_runs/                # Polished final showcase notebooks
  experiments/               # Hyperparameter tuning iteration notebooks (iters 1–24)
docs/                        # Architecture notes, workflow docs, tuning history
artifacts/                   # Training outputs (checkpoints, plots, videos) — gitignored
```

**Key entry points for reviewers:**

- **Final results:** `notebooks/final_runs/` — the showcase training notebooks
- **Source code:** `src/tiny_dreamer_highway/` — the complete Dreamer pipeline
- **Configuration:** `notebooks/configs/final_run_iter17_score.yaml` — the winning configuration (iter 17)
- **Tests:** `tests/` — the full test suite
- **Tuning history:** `docs/config_tuning_history.md` — all 24 iterations documented
