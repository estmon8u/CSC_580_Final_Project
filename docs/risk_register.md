# Risk Register

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Highway-Env setup differs on Windows | Environment fails to launch | Keep imports lazy and validate with a tiny smoke script |
| Dreamer training is unstable | Slow or failed convergence | Build module-by-module tests before integration |
| Videos and checkpoints become large | Repository clutter | Save outputs under `artifacts/` and ignore generated binaries |
| End-to-end tests are too slow | Feedback loop degrades | Use fast unit tests and only tiny smoke integration |
| Submission package drifts from final instructions | Missing deliverables or grading penalties | Track report, notebook-cleanup, Drive-folder, and artifact-bundle requirements in the checklist before submission |
