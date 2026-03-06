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
6. install the package with editable mode if needed
7. run a small sanity pass before expensive cells

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
