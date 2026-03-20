# Config Tuning History — Tiny Dreamer Highway

> **Purpose:** Track every configuration iteration, what changed, why, and what
> behavior resulted.  Prevents accidentally revisiting failed settings.
>
> Last updated: 2026-03-19

---

## Table of Contents

1. [Iteration 0 — Original Code Defaults](#iteration-0--original-code-defaults)
2. [Iteration 1 — Tuned Baseline (Current)](#iteration-1--tuned-baseline-current)
3. [Key Discoveries](#key-discoveries)
4. [Parameters NOT to Change](#parameters-not-to-change)

---

## Iteration 0 — Original Code Defaults

**Source:** `config.py` Pydantic defaults (what you get if YAML doesn't override).

| Parameter | Value |
|---|---|
| `actor_init_std` | **5.0** |
| `vehicles_count` | 50 |
| `npc_speed_scale` | 1.0 |
| `smoothing_factor` | 0.6 |
| `lateral_scale` | 0.35 |
| `longitudinal_scale` | 1.0 |
| `high_speed_reward` | 0.4 |
| `right_lane_reward` | 0.1 |
| `lane_change_reward` | 0.0 |
| `overtake_reward` | 0.0 |
| `normalize_reward` | true |
| `reward_speed_range` | [20.0, 30.0] |
| `offroad_terminal` | true |
| `offroad_penalty` | 3.0 |
| `steering_penalty` | 0.05 |
| `steering_change_penalty` | 0.1 |
| `discount` | 0.99 |
| `max_episode_steps` | 40 |
| `frame_stack` | 1 |
| `imagination_horizon` | 5 |
| `batch_size` | 4 |

**Result:** Agent trailed behind traffic — passive following behavior.

---

## Iteration 1 — Tuned Baseline (Current)

**What we were trying to achieve:** Stable continuous-control baseline on GPU
with reward shaping to encourage faster driving and overtaking.

**Source:** `final_run.yaml` / `training_run.yaml` / `final_showcase_eval.yaml`
(all three synced).

| Parameter | Value | vs. Defaults |
|---|---|---|
| `actor_init_std` | **1.0** | ↓ from 5.0 |
| `vehicles_count` | 28 | ↓ from 50 |
| `npc_speed_scale` | 0.85 | ↓ from 1.0 |
| `smoothing_factor` | 0.6 | same |
| `lateral_scale` | 0.35 | same |
| `longitudinal_scale` | 0.7 | ↓ from 1.0 |
| `high_speed_reward` | 0.8 | ↑ from 0.4 |
| `right_lane_reward` | 0.0 | ↓ from 0.1 |
| `lane_change_reward` | 0.0 | same |
| `overtake_reward` | 2.5 | ↑ from 0.0 |
| `normalize_reward` | true | same |
| `reward_speed_range` | [26.0, 35.0] | shifted up |
| `offroad_terminal` | true | same |
| `offroad_penalty` | 0.5 | ↓ from 3.0 |
| `steering_penalty` | 0.0075 | ↓ from 0.05 |
| `steering_change_penalty` | 0.02 | ↓ from 0.1 |
| `discount` | 0.99 | same |
| `max_episode_steps` | 75 | ↑ from 40 |
| `frame_stack` | 1 | same |
| `imagination_horizon` | 20 | ↑ from 5 |
| `batch_size` | 256 | ↑ from 4 |
| `cycles` | 2000 | ↑ from 10 |
| `warm_start_steps` | 10000 | ↑ from 64 |
| `policy_steps` | 32 | ↑ from 8 |
| `use_amp` | true | new |
| `amp_dtype` | bfloat16 | new |
| `world_model_updates_per_cycle` | 8 | ↑ from 1 |
| `behavior_updates_per_cycle` | 8 | ↑ from 1 |

**Observed result:** Agent learned to **trail behind traffic** — stayed in lane,
drove safely, avoided risk, and did not reliably overtake.

**Why these values were chosen:**
- `actor_init_std: 1.0` — `softplus(1.0) ≈ 1.31`, good exploration range without
  tanh saturation. The original 5.0 caused bang-bang control (see Key Discoveries).
- `vehicles_count: 28` and `npc_speed_scale: 0.85` — moderately easier road
  without being an empty highway.
- `high_speed_reward: 0.8` and `reward_speed_range: [26, 35]` — encourage faster
  driving while preserving the upper-speed target.
- `overtake_reward: 2.5` — meaningful incentive to pass.
- `offroad_penalty: 0.5`, `steering_penalty: 0.0075`, and
  `steering_change_penalty: 0.02` — preserve smooth driving without
  over-punishing lane changes.

---

## Key Discoveries

### 1. `actor_init_std: 5.0` causes tanh saturation (bang-bang)
- `std = softplus(raw_std + init_std) + min_std`
- `softplus(5.0) ≈ 5.0` → noise so large that tanh squashes everything to ±1
- Agent can only do hard-left/hard-right/full-gas/full-brake
- **Fix:** `actor_init_std: 1.0` → `softplus(1.0) ≈ 1.31` → smooth exploration

### 2. `normalize_reward: true` has two hidden effects
- **highway-env side:** Maps raw rewards to [0, 1], giving a free +0.4 baseline
  just for existing (not crashing)
- **Wrapper side:** `_additive_scale = 1/raw_span` multiplies all
  penalties/bonuses by ~0.4, making them 2.5× weaker than configured
- **Caution:** If `normalize_reward` is on, configured penalties must be scaled
  up ~2.5× to have their intended effect

### 3. `offroad_terminal: false` enables the spinning exploit
- With continuous control, the agent can steer off-road and spin indefinitely
- Combined with the +0.4 existence baseline from normalize_reward, spinning is a
  "safe" local optimum
- **Rule:** Always keep `offroad_terminal: true` with continuous control

### 4. Trailing behind traffic is rational behavior
- If collision penalty is high and overtaking is risky, the expected value of
  staying behind is higher than the expected value of passing
- **Fix:** Make overtaking easier (fewer/slower NPCs) AND more rewarding
  (high `overtake_reward`), NOT by reducing collision penalty

### 5. `smoothing_factor` tradeoff
- High (0.6): Very stable but sluggish — can't react in time to overtake
- Low (0.0): Fully responsive but jittery — policy noise maps directly to action
- **Sweet spot:** 0.1–0.2 for continuous control with lane changes

### 6. No entropy bonus in actor loss
- The actor loss is purely `-weighted_mean(returns, weights)` — no entropy term
- Once the actor's learned std collapses (which happens fast when trailing is
  safest), there's nothing forcing it to explore lane changes
- This creates a feedback loop: trailing data → world model only knows trailing
  → imagined overtakes look bad → policy stays trailing

---

## Parameters NOT to Change

These were evaluated and deliberately rejected:

| Suggestion | Why Rejected |
|---|---|
| **Trailing penalty** (negative reward for being behind NPCs) | "Kamikaze Trap" — punishes the agent for being near traffic at all, encourages suicidal passes |
| **Non-terminal collisions** (`collision_terminal: false`) | Creates "zombie states" — agent keeps collecting penalties after crashing, poisons the replay buffer with meaningless transitions |
| **`smoothing_factor: 0.0`** | Too jittery; policy noise translates directly to action oscillation. 0.1–0.2 is the floor. |
| **Reducing collision penalty** | Makes crashes cheap — agent learns to ram through traffic instead of passing cleanly |

---

## Config Diff Quick-Reference

| Parameter | Iter 0 | **Iter 1 (NOW)** |
|---|---|---|
| `actor_init_std` | 5.0 | **1.0** |
| `vehicles_count` | 50 | **28** |
| `npc_speed_scale` | 1.0 | **0.85** |
| `smoothing_factor` | 0.6 | **0.6** |
| `lateral_scale` | 0.35 | **0.35** |
| `longitudinal_scale` | 1.0 | **0.7** |
| `high_speed_reward` | 0.4 | **0.8** |
| `overtake_reward` | 0.0 | **2.5** |
| `normalize_reward` | true | **true** |
| `reward_speed_range` | [20,30] | **[26,35]** |
| `offroad_terminal` | true | **true** |
| `offroad_penalty` | 3.0 | **0.5** |
| `steering_penalty` | 0.05 | **0.0075** |
| `steering_change_penalty` | 0.1 | **0.02** |
| `discount` | 0.99 | **0.99** |
| `max_episode_steps` | 40 | **75** |
| `frame_stack` | 1 | **1** |
| `lane_change_reward` | 0.0 | **0.0** |
| **Goal** | Raw defaults | Stable GPU baseline |
| **Behavior** | Trailing | Trailing |
