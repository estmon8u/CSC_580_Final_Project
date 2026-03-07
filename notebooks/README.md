# Notebook Assets

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

Use notebooks in this folder for polished experiments, debugging views, and final presentation assets.

Current notebook set:

- `01_colab_setup_and_smoke_tests.ipynb` for Colab environment setup plus end-to-end smoke validation against pushed package code
- `02_colab_sanity_run.ipynb` for a short runner validation job
- `03_colab_baseline_run.ipynb` for the first real baseline comparison run
- `04_colab_h100_run.ipynb` for the larger H100-oriented run after the baseline is stable
- `05_colab_optimized_run.ipynb` for the optimized comparison run (AdamW, grad clipping, LR warm-up)

Suggested order:

1. run `01_colab_setup_and_smoke_tests.ipynb` to validate the pushed code path
2. run `02_colab_sanity_run.ipynb` to confirm the training runner, checkpoints, logs, and analysis flow
3. run `03_colab_baseline_run.ipynb` for the first real comparison-quality baseline
4. run `05_colab_optimized_run.ipynb` for the optimized comparison against the baseline
5. run `04_colab_h100_run.ipynb` only after the baseline and optimized runs are stable
The source of truth stays in `src/tiny_dreamer_highway/`.
