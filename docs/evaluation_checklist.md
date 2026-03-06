# Evaluation Checklist

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

## Early checks

- config loads from YAML
- replay buffer stores transitions correctly
- sampled sequences have expected shapes
- Highway-Env factory returns an RGB-compatible environment config

## Mid-stage checks

- world model forward pass reconstructs images
- reward prediction loss decreases on a tiny batch
- imagined latent rollouts maintain valid shapes

## Final checks

- policy acts in `highway-v0`
- n-step predictions can be compared with actual frames
- plots and videos are saved into `artifacts/`
