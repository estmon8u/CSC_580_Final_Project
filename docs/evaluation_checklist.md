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
- comparison media can be reused in the final submission bundle
- a manifest and zip bundle can be exported for submission handoff

## Final report polish

- select one representative metrics plot for the write-up
- select one representative comparison image and one short GIF for the write-up
- confirm the bundle manifest matches the final artifact set cited in the report

## Submission requirements

- the final Google Drive submission folder includes code, notebook, videos, and report
- the notebook is cleaned for presentation quality with no draft cells or stray debugging output
- every submitted file includes name, course, assignment, and AI-tool attribution at the top
- the report states the word count near the top and reaches at least 2,500 words
- the report includes charts, tables, deep analysis, conclusions, and course reflection
