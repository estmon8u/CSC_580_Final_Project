# Colab Git + Drive Workflow

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

## Source of truth

Use the GitHub repository as the source of truth for code and project documents.

Use Google Drive for:

- checkpoints
- replay snapshots
- plots
- videos
- submission bundles
- exported notebook results

## Update loop

1. modify code locally in VS Code
2. Create test cases for the new code or changes
3. run `pytest` for all new and existing tests to validate the changes don't break existing functionality
4. commit and push only after tests pass
5. open Colab and mount Google Drive
6. clone or pull the repository into the Colab workspace
7. install the package with `python -m pip install -e .`
8. run a small sanity pass before expensive cells

## Dependency management

Use [pyproject.toml](../pyproject.toml) as the dependency source of truth.

The notebook should not duplicate the full dependency list. Keep notebook install steps thin:

- install from the repository root
- print versions for quick validation
- restart the runtime only if Colab had to replace a preloaded package

The current dependency floor is chosen to stay compatible with Colab's default NumPy 2 ecosystem rather than forcing an older NumPy 1.x stack into the runtime.

## Colab sanity pass

Before training, always verify:

- config file loads
- package imports succeed
- environment can reset
- one random action step succeeds
- artifact directory exists on Drive

Keep Colab notebooks split by purpose:

- one notebook for setup and smoke validation
- one notebook for real or fake training runs and checkpoint generation

## Replay sequence guidance

Dreamer world-model updates train on contiguous replay sequences. That means replay readiness depends on episode boundaries, not only on raw buffer size.

- if `sequence_length > max_episode_steps`, training is impossible and should be treated as a configuration error
- if `offroad_terminal=true`, random warm-start rollouts may end before enough length accumulates for a valid sequence window
- for short validation jobs, prefer smaller `sequence_length`, smaller batch sizes, and more conservative terminal settings
- for real runs, keep notebook overrides close to the YAML defaults unless you intentionally want a reduced-data experiment

The core trainer now auto-collects extra random transitions when the requested warm start does not yet yield enough valid sequences. That makes notebook overrides less brittle, but it does not replace sensible run settings.

## Drive layout suggestion

Use a Drive folder like:

- `MyDrive/CSC_580_Final_Project/artifacts/checkpoints`
- `MyDrive/CSC_580_Final_Project/artifacts/media`
- `MyDrive/CSC_580_Final_Project/artifacts/logs`
- `MyDrive/CSC_580_Final_Project/artifacts/bundles`

## Rule for this project

Every code change should be tested locally before pushing. Colab should pull validated changes rather than serving as the primary editing environment.
