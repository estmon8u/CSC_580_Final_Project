# Risk Register

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Highway-Env setup differs on Windows | Environment fails to launch | Keep imports lazy and validate with a tiny smoke script |
| Dreamer training is unstable | Slow or failed convergence | Build module-by-module tests before integration |
| Value hallucination (imagined values diverge from true returns) | Agent learns no useful policy | Corrected TD(λ) formula to use next-state values; validated against DreamerV1 reference |
| Short-horizon world model looks good but drifts badly over multiple steps | Actor learns from unrealistic imagined futures | Use observation NLL, latent rollout consistency metrics, and overshooting KL regularization |
| Narrow actor exploration early in training | Policy collapses to deterministic before learning | Actor uses init_std=5.0, mean_scale=5.0 with TanhTransform Jacobian correction |
| Undersized model capacity vs reference | World model too weak to capture environment dynamics | ModelConfig defaults match DreamerV1 reference (1024 embedding, 200+30 latent) |
| Videos and checkpoints become large | Repository clutter | Save outputs under `artifacts/` and ignore generated binaries |
| End-to-end tests are too slow | Feedback loop degrades | Use fast unit tests with small model dims and only tiny smoke integration |
| Submission package drifts from final instructions | Missing deliverables or grading penalties | Track report, notebook-cleanup, Drive-folder, and artifact-bundle requirements in the checklist before submission |
| H100 runtime exhaustion before convergence | Wasted compute budget | Use screening config (h100_screening) with staged checkpoints and early stopping |
