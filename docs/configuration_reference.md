# Configuration Reference

**Name:** Esteban  
**Course:** CSC 580 AI 2  
**Assignment:** Final Project — Dream the Road  
**AI tools consulted:** GitHub Copilot

All experiment settings are defined in a single YAML file and parsed into
a typed Pydantic model (`ExperimentConfig`). This document describes every
field, its default value, and how it affects training.

Example YAML files live in `examples/`.

---

## Top-level

| Setting | Default | Purpose |
|---|---|---|
| `seed` | `7` | Random seed for NumPy, PyTorch, and the environment. Ensures reproducibility — the same seed produces the same initial weights, random exploration, and environment resets. |
| `device` | `cpu` | PyTorch device for all tensors and models. Use `cuda` on Colab GPU runtimes, `cpu` for local testing. |

---

## `env` — Environment Configuration

| Setting | Default | Purpose |
|---|---|---|
| `env_id` | `highway-v0` | Which Gymnasium environment to create. `highway-v0` is a multi-lane highway driving simulation from the Highway-Env library. |
| `observation_height` | `64` | Height (in pixels) of the grayscale image observation fed to the CNN encoder. Smaller = faster but less detail. Range: 32–256. |
| `observation_width` | `64` | Width (in pixels). Kept equal to height for a square input that the CNN can process uniformly. Range: 32–256. |
| `frame_stack` | `1` | Number of consecutive frames stacked into the observation's channel dimension. `1` = single frame (the RSSM handles temporal memory instead). Values >1 give the encoder explicit motion information. Range: 1–4. |
| `max_episode_steps` | `40` | Hard limit on how many environment steps one episode can last. Prevents the agent from driving forever and caps per-episode data collection. Range: 10–500. |

### `env.action` — Action Space

| Setting | Default | Purpose |
|---|---|---|
| `type` | `continuous` | Action space type. `continuous` means the actor outputs real-valued throttle/steering instead of discrete lane-change commands. |
| `longitudinal` | `true` | Whether the agent controls acceleration/braking. If `false`, the car drives at a fixed speed. |
| `lateral` | `true` | Whether the agent controls steering. If `false`, no lane changes. |
| `longitudinal_scale` | `1.0` | Multiplier applied to the raw longitudinal action before sending it to the environment. `1.0` = full throttle range. Lower values (e.g., `0.5`) limit max acceleration, making driving smoother but slower. Range: (0, 1]. |
| `lateral_scale` | `0.35` | Multiplier on steering. `0.35` compresses the steering range to ~35% of maximum, preventing wild swerving. This is the single biggest stabilizer for smooth driving. Range: (0, 1]. |
| `smoothing_factor` | `0.6` | Exponential moving average coefficient between consecutive actions: `a_t = α · a_{t-1} + (1 − α) · a_raw`. Higher = more inertia, smoother but less responsive. `0.0` = no smoothing. Range: [0, 1). |

### `env.reward` — Reward Shaping

| Setting | Default | Purpose |
|---|---|---|
| `collision_reward` | `-1.0` | Reward given on crash (native Highway-Env setting). Negative = punishment. This is the primary safety signal. |
| `right_lane_reward` | `0.1` | Bonus for being in the rightmost lane. Encourages orderly driving. |
| `high_speed_reward` | `0.4` | Bonus for driving within the target speed range. The biggest positive reward — drives the agent to maintain speed. |
| `lane_change_reward` | `0.0` | Reward/penalty for changing lanes. `0.0` = neutral. Set negative to discourage unnecessary weaving. |
| `normalize_reward` | `true` | Whether Highway-Env normalizes reward to approximately [−1, 1]. Helps stabilize gradient magnitudes during training. |
| `reward_speed_range` | `[20.0, 30.0]` | The speed window (m/s) over which `high_speed_reward` is linearly interpolated. Below 20 → 0 bonus, at 30 → full bonus. |
| `offroad_terminal` | `true` | If `true`, going off-road **ends the episode** immediately (like a crash). Gives a very strong learning signal: "don't leave the road." If `false`, the agent gets penalized but the episode continues. |
| `offroad_penalty` | `3.0` | Flat penalty (subtracted from reward) every step the car is off the road surface. Larger values make the agent strongly avoid lane departure. Applied by our custom `DrivingPenaltyRewardWrapper`. |
| `steering_penalty` | `0.05` | Per-step cost proportional to |steering|: `-0.05 × |a_lateral|`. Discourages gratuitous steering even when on-road. |
| `steering_change_penalty` | `0.1` | Per-step cost proportional to |Δsteering|: `-0.1 × |a_t − a_{t-1}|`. Penalizes jerky direction changes, encouraging smooth trajectories. |

---

## `replay` — Experience Replay Buffer

| Setting | Default | Purpose |
|---|---|---|
| `capacity` | `10000` | Maximum transitions stored. Older data is overwritten when full. Larger = more diverse training data but more RAM. H100 config uses 100k to exploit GPU memory. Range: ≥128. |
| `sequence_length` | `8` | Number of consecutive time-steps sampled as one training sequence. The RSSM processes these sequentially to learn temporal dynamics. Longer sequences capture longer-horizon patterns but use more memory. Range: 2–128. |
| `batch_size` | `4` | Number of sequences per training batch when sampling from replay. (In practice this is usually overridden by `training.batch_size`.) Range: 1–512. |

---

## `training` — Training Loop

| Setting | Default | Purpose |
|---|---|---|
| `batch_size` | `4` | Number of sequence batches drawn from replay per gradient step. `4` for CPU sanity runs, `256` to saturate H100 memory bandwidth. Range: 1–512. |
| `imagination_horizon` | `5` | How many future steps the actor/critic "imagine" using the world model during behaviour learning. Longer = better credit assignment, but compounds model errors. Range: 2–64. |
| `world_model_lr` | `3e-4` | Learning rate for the world model optimizer (encoder + RSSM + decoder + reward predictor). 3e-4 is a standard DreamerV1 value. |
| `actor_lr` | `8e-5` | Learning rate for the actor (policy network). Lower than the world model LR because policy updates depend on world-model gradients — too fast and it chases a moving target. |
| `critic_lr` | `8e-5` | Learning rate for the critic (value network). Matched to actor LR for stability. |
| `kl_weight` | `1.0` | Coefficient on the KL-divergence term in the world model loss: `L = recon + reward + β · D_KL`. Controls how much the latent distribution is regularized. Higher = more compressed latent space. |
| `free_nats` | `3.0` | KL "free bits" threshold. KL values below 3.0 nats are zeroed, preventing the model from over-regularizing when the latent is already compact. Standard DreamerV1 trick. |
| `grad_clip_norm` | `100.0` | Maximum L2 norm for gradient clipping. Prevents exploding gradients during early training. `100.0` is permissive — tighten to `10.0`–`50.0` if you see NaNs. Range: (0, 10000]. |
| `lr_warmup_steps` | `0` | Number of optimizer steps over which the learning rate linearly ramps from 0 to the configured LR. Prevents large early gradients from destabilizing fresh random weights. `0` = no warmup. Range: 0–10000. |
| `use_amp` | `false` | Enable Automatic Mixed Precision. When `true`, forward passes run in lower precision (bf16/fp16), roughly doubling throughput on tensor-core GPUs (A100, H100). |
| `amp_dtype` | `bfloat16` | Which reduced-precision dtype to use with AMP. `bfloat16` is preferred on H100 (native tensor core dtype, no loss scaling needed). `float16` works on older GPUs but requires a GradScaler. Allowed: `bfloat16`, `float16`. |
| `use_flash_optimizer` | `false` | Whether to try `flashoptim.FlashAdamW`, a fused optimizer that keeps master weights in bf16 (saving ~50% optimizer memory). Falls back to standard `AdamW` if unavailable (e.g., on CPU or without the package). |
| `world_model_updates_per_cycle` | `1` | Gradient steps on the world model per data-collection cycle. `1` = one update per cycle (conservative). `16` = train harder on each batch of collected data, keeping the GPU busy between environment steps. Range: 1–256. |
| `behavior_updates_per_cycle` | `1` | Gradient steps on the actor + critic per cycle. Same tradeoff — more updates = faster learning but risk overfitting to current replay data. Range: 1–256. |
| `cycles` | `10` | Total training cycles (outer loop iterations). Each cycle = collect data → train world model → train actor/critic → checkpoint. Sanity runs use 10; real experiments 500+. Range: 1–1,000,000. |
| `warm_start_steps` | `64` | Number of random-policy environment steps collected **before** any training begins. Fills the replay buffer with diverse initial data so the first gradient steps see varied transitions. Must be ≥ `batch_size × sequence_length`. Range: 0–1,000,000. |
| `policy_steps` | `8` | Environment steps collected per cycle using the **current policy** (not random). More steps = more fresh on-policy data per cycle but slower iteration. Range: 0–1,000,000. |
| `checkpoint_interval` | `5` | Save a model checkpoint every N cycles. Lower = more frequent saves (safer against crashes) but more disk usage. Range: 1–1,000,000. |

---

## `model` — Model Architecture Dimensions

All model dimension defaults match the open-source DreamerV1 reference implementation.
These are parsed into `ModelConfig` inside `ExperimentConfig`.

| Setting | Default | Purpose |
|---|---|---|
| `embedding_dim` | `1024` | Output dimension of the CNN encoder. The encoder maps 64×64 grayscale images to a flat vector of this size. Larger = richer visual features at the cost of more parameters. Range: 32–4096. |
| `deterministic_dim` | `200` | Size of the GRU hidden state in the RSSM. This is the deterministic part of the latent state. Together with `stochastic_dim`, it forms the full latent state used by the actor and critic. Range: 32–2048. |
| `stochastic_dim` | `30` | Size of the stochastic latent variable sampled by the RSSM posterior/prior. Smaller than the deterministic part — captures irreducible uncertainty. Range: 8–512. |
| `hidden_dim` | `200` | Hidden layer width for the RSSM prior and posterior MLPs. Range: 32–2048. |
| `rssm_num_layers` | `2` | Number of hidden layers in the RSSM prior and posterior networks. `2` matches the DreamerV1 reference. Range: 1–4. |
| `actor_hidden_dim` | `200` | Hidden layer width for the actor (policy) network. Range: 32–2048. |
| `actor_num_layers` | `2` | Number of hidden layers in the actor MLP. Range: 1–4. |
| `critic_hidden_dim` | `200` | Hidden layer width for the critic (value) network. Range: 32–2048. |
| `critic_num_layers` | `3` | Number of hidden layers in the critic MLP. `3` matches DreamerV1. Range: 1–6. |
| `reward_hidden_dim` | `200` | Hidden layer width for the reward predictor head. Range: 32–2048. |
| `reward_num_layers` | `2` | Number of hidden layers in the reward predictor. Range: 1–4. |

The full latent dimension seen by actor, critic, and decoder is `deterministic_dim + stochastic_dim` (default: 230).

---

## Training Cycle Overview

A single training cycle executes the following steps in order:

```
1. Collect `policy_steps` environment transitions using the actor
2. Store them in the replay buffer (size = `capacity`)
3. Sample `batch_size` sequences of length `sequence_length`
4. Run `world_model_updates_per_cycle` gradient steps on the world model
   - Collect posterior latent states from each WM training batch
5. Pass WM posteriors as imagination start states for actor/critic
6. Imagine `imagination_horizon` steps forward for actor/critic
7. Run `behavior_updates_per_cycle` gradient steps on actor + critic
8. Every `checkpoint_interval` cycles, save weights to disk
```

---

## Example Profiles

| Profile | YAML | Batch | Seq Len | Cycles | AMP | Model | Notes |
|---|---|---|---|---|---|---|---|
| CPU sanity | `base_experiment.yaml` | 4 | 8 | 10 | off | small (128+32) | Quick smoke test on any machine |
| Colab production | `training_run.yaml` | 32 | 32 | 500 | off | reference | Real training on T4 GPU |
| Optimized | `optimized_experiment.yaml` | 32 | 32 | 500 | off | reference | AdamW + grad clip + LR warmup |
| H100 full | `h100_experiment.yaml` | 128 | 32 | 500 | off | reference | Large-scale GPU run |
| H100 + AMP | `h100_amp_experiment.yaml` | 256 | 32 | 2000 | bf16 | reference | Maximum H100 throughput with FlashAdamW |
| H100 screening | `h100_screening_experiment.yaml` | 32 | 32 | 800 | bf16 | reference | Safer screening: 24 WM updates, 4 behavior updates |
