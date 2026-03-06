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
- exported notebook results

## Update loop

1. modify code locally in VS Code
2. run `pytest`
3. commit and push only after tests pass
4. open Colab and mount Google Drive
5. clone or pull the repository into the Colab workspace
6. install the package with `python -m pip install -e .`
7. run a small sanity pass before expensive cells

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

## Drive layout suggestion

Use a Drive folder like:

- `MyDrive/CSC_580_Final_Project/artifacts/checkpoints`
- `MyDrive/CSC_580_Final_Project/artifacts/media`
- `MyDrive/CSC_580_Final_Project/artifacts/logs`

## Rule for this project

Every code change should be tested locally before pushing. Colab should pull validated changes rather than serving as the primary editing environment.
