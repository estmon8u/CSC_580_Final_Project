# Config Tuning History — Tiny Dreamer Highway

> **Purpose:** Track every configuration iteration, what changed, why, and what
> behavior resulted.  Prevents accidentally revisiting failed settings.
>
> Last updated: 2026-03-20 (Iter 17-19 complete, Phase 4 iters 20-23)

---

## Table of Contents

1. [Iteration 0 — Tuned Baseline (Starting Point)](#iteration-0--tuned-baseline-starting-point)
2. [Isolation Plan — One Knob at a Time (Iters 1-5)](#isolation-plan--one-knob-at-a-time-iters-1-5)
3. [Cumulative Results — Iters 1-5](#cumulative-results--iters-1-5)
4. [Isolation Plan — Phase 2 (Iters 6-10)](#isolation-plan--phase-2-iters-6-10)
5. [Phase 2 Results — Iters 6-10](#phase-2-results--iters-6-10)
6. [Isolation Plan — Phase 3 (Iters 12-16)](#isolation-plan--phase-3-iters-12-16)
7. [Phase 3 Results — Iters 12-16](#phase-3-results--iters-12-16)
8. [Iter 17 — Combined 13+15](#iter-17--combined-1315)
9. [Iter 18 — Entropy Regularization (code change)](#iter-18--entropy-regularization-code-change)
10. [Iter 19 — Iter 17 + Entropy](#iter-19--iter-17--entropy)
11. [Phase 4 — Refinement (Iters 20-23)](#phase-4--refinement-iters-20-23)
12. [Key Discoveries](#key-discoveries)
13. [Parameters NOT to Change](#parameters-not-to-change)

---

## Iteration 0 — Tuned Baseline (Starting Point)

**What we were trying to achieve:** Stable continuous-control baseline on GPU
with reward shaping to encourage faster driving and overtaking.

**Source:** `final_run.yaml` / `training_run.yaml` / `final_showcase_eval.yaml`
(all three synced).

| Parameter | Value |
|---|---|
| `actor_init_std` | 1.0 |
| `vehicles_count` | 28 |
| `npc_speed_scale` | 0.85 |
| `smoothing_factor` | 0.6 |
| `lateral_scale` | 0.35 |
| `longitudinal_scale` | 0.7 |
| `high_speed_reward` | 0.8 |
| `right_lane_reward` | 0.0 |
| `lane_change_reward` | 0.0 |
| `overtake_reward` | 2.5 |
| `normalize_reward` | true |
| `reward_speed_range` | [26.0, 35.0] |
| `offroad_terminal` | true |
| `offroad_penalty` | 0.5 |
| `steering_penalty` | 0.0075 |
| `steering_change_penalty` | 0.02 |
| `discount` | 0.99 |
| `max_episode_steps` | 75 |
| `frame_stack` | 1 |
| `imagination_horizon` | 20 |
| `batch_size` | 256 |
| `cycles` | 2000 |
| `warm_start_steps` | 10000 |
| `policy_steps` | 32 |
| `use_amp` | true |
| `amp_dtype` | bfloat16 |
| `world_model_updates_per_cycle` | 8 |
| `behavior_updates_per_cycle` | 8 |

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

## Isolation Plan — One Knob at a Time (Iters 1-5)

> **Method:** Each iteration changes exactly ONE parameter from Iter 0.
> Changes are cumulative — if Iter 1 doesn't fix trailing, Iter 2 adds its
> change on top of Iter 1's change, and so on.  Stop when overtaking appears.
>
> **Diagnosis:** Iter 0 trails because the agent deems crashing as worst-case
> and the expected value of staying behind exceeds the expected value of passing.

### Iter 1 — `smoothing_factor: 0.6 → 0.2`

**Hypothesis:** Physical capability bottleneck. With 0.6, each action is 60%
the previous action — the agent can't complete a lane change before conditions
change. Even a perfectly trained policy would struggle to overtake.

**What success looks like:** Agent attempts lane changes (even if it crashes).
**What failure looks like:** Still trails → problem is reward/risk, not control.

### Iter 2 — `vehicles_count: 28 → 12`

**Hypothesis:** Too many cars = every lane has traffic = every pass attempt
risks collision. Fewer cars means more open lanes and lower collision
probability per overtake attempt.

**What success looks like:** Agent finds gaps and passes slower traffic.
**What failure looks like:** Still trails on a near-empty road → doesn't value speed.

### Iter 3 — `npc_speed_scale: 0.85 → 0.65`

**Hypothesis:** NPCs at 0.85× (~21-25 m/s) are close to ego speed. Bigger
speed differential (~16-19 m/s NPCs) makes overtaking happen naturally during
random exploration, giving the world model real pass data to learn from.

**What success looks like:** Agent drives past crawling NPCs without hesitation.
**What failure looks like:** Still trails behind 16 m/s NPCs → reward signal bottleneck.

### Iter 4 — `normalize_reward: true → false`

**Hypothesis:** Normalization gives a free +0.4 baseline for existing and
dilutes all penalties/bonuses by 0.4×. Trailing earns ~0.4 per step for free,
so the marginal value of overtaking is suppressed.

**What success looks like:** Rewards are sharper — agent differentiates good/bad states.
**What failure looks like:** Off-road/spinning regression → needs `offroad_penalty` bump.

### Iter 5 — `frame_stack: 1 → 3`

**Hypothesis:** Single frame = no temporal info. Agent can't tell if a gap is
opening or closing, so it can't time a lane change. Frame stacking gives
implicit velocity/acceleration from pixel differences.

**What success looks like:** Better-timed passes; fewer collisions during overtakes.
**What failure looks like:** Still trails → temporal info isn't the bottleneck.

---

## Cumulative Results — Iters 1-5

The cumulative run (iter5_framestack = all 5 changes stacked) was the winner.
Trained to ~1000 cycles from scratch, then extended with `smoothing_factor`
adjusted from 0.2 → 0.5.

| Parameter | Iter 0 | Iter 5 cumulative |
|---|---|---|
| `smoothing_factor` | 0.6 | 0.5 (was 0.2, raised to fix wobble) |
| `vehicles_count` | 28 | 12 |
| `npc_speed_scale` | 0.85 | 0.65 |
| `normalize_reward` | true | false |
| `frame_stack` | 1 | 3 |

**Observed behavior (positive):**

- Agent **no longer trails** behind traffic — major breakthrough
- Agent is **sometimes traffic-aware** — detects and weaves around NPC vehicles
- Agent **sometimes overtakes** — actually passes slower NPCs
- `world_total` tripled (expected — frame_stack=3 means 3× pixels in reconstruction loss)

**Observed behavior (problems):**

- **Too fast → crashing:** Agent floors it and can't react to traffic in time
- **Rapid L/R jitter:** Actor outputs high-frequency left/right oscillation that
  cancels out — moves roughly straight but with constant steering noise
- **Inconsistent:** Traffic awareness and overtaking are intermittent, not reliable

**Smoothing factor journey:**

- 0.6 (Iter 0): Over-damped → couldn't complete lane changes → trailing
- 0.2 (Iter 5 original): Under-damped → wobble, steering noise passes through
- 0.5 (current): Compromise — responsive enough but less jittery

**Checkpoint saved:** `checkpoint_0025.pt` (warm_start=10000 + 25 cycles) used
as resume point for all future iterations. Frame_stack=3 is baked in.

---

## Isolation Plan — Phase 2 (Iters 6-10)

> **Base config:** Iter 5 cumulative (smoothing=0.5, vehicles=12, npc_speed=0.65,
> normalize=false, frame_stack=3)
>
> **Method:** Each iteration changes ONE parameter from this base. NOT cumulative —
> clean attribution per knob. All resume from `checkpoint_0025.pt`.
>
> **Two problems to fix:**
>
> 1. Too fast → crashing (speed control)
> 2. Rapid L/R jitter (steering oscillation)

### Iter 6 — `steering_change_penalty: 0.02 → 0.08`

**Targets:** L/R jitter  
**Hypothesis:** Rapid L/R oscillation = high ∆steering per step. A 4× increase
in the change penalty directly taxes this behavior, incentivizing the agent to
hold a steady heading and only steer when committed to a lane change.

**What success looks like:** Smooth lane-hold; deliberate, committed lane changes.  
**What failure looks like:** Agent stops steering entirely → over-penalized.

### Iter 7 — `lateral_scale: 0.35 → 0.15`

**Targets:** L/R jitter  
**Hypothesis:** Even when actor outputs ±1 steering, the physical effect is
capped at 0.15 instead of 0.35. Jitter amplitude shrinks and lane changes
become more gradual.

**What success looks like:** Smoother trajectories; agent still manages lane changes.  
**What failure looks like:** Can't complete lane changes in time → reverts to trailing.

### Iter 8 — `reward_speed_range: [26,35] → [22,30]`

**Targets:** Too fast → crashing  
**Hypothesis:** Current range incentivizes 35 m/s which is too fast to react to
traffic. Lowering to [22,30] still rewards overtaking slower NPCs (~16-19 m/s)
without incentivizing dangerous speed.

**What success looks like:** Agent cruises at ~28-30 m/s; fewer high-speed crashes.  
**What failure looks like:** Agent lacks urgency to overtake.

### Iter 9 — `high_speed_reward: 0.8 → 0.3`

**Targets:** Too fast → crashing  
**Hypothesis:** Speed reward dominates the reward signal. Reducing the weight
from 0.8 to 0.3 makes collision/steering penalties relatively more important,
so the agent values survival over speed.

**What success looks like:** Still prefers faster, but doesn't kamikaze for speed.  
**What failure looks like:** Not enough speed motivation → trails again.

### Iter 10 — `steering_penalty: 0.0075 → 0.025`

**Targets:** L/R jitter  
**Hypothesis:** Penalizing steering magnitude (not just changes) pushes the agent
toward straight-line driving by default. Lane changes only happen when the
overtake reward clearly outweighs the steering cost.

**What success looks like:** Small, deliberate turns; no constant micro-adjustments.  
**What failure looks like:** Afraid to steer → trails or drifts off-road.

---

## Phase 2 Results — Iters 6-10

All ran 500 cycles (250 + 250 extension) from `checkpoint_00025.pt` (iter5 base).

| Iter | Knob | Result |
|---|---|---|
| 6 | `steering_change_penalty: 0.02 → 0.08` | **Rejected** — stopped avoiding traffic. Over-penalized direction changes killed lane-change ability. |
| 7 | `lateral_scale: 0.35 → 0.15` | **Rejected** — stopped avoiding traffic. Physical steering cap too low to complete lane changes. |
| 8 | `reward_speed_range: [26,35] → [22,30]` | **Good for speed** — reduced high-speed crashes. Agent cruises at more manageable speeds. |
| 9 | `high_speed_reward: 0.8 → 0.3` | **Good for speed** — less kamikaze behavior. Agent still overtakes but doesn't floor it. |
| 10 | `steering_penalty: 0.0075 → 0.025` | **Best overall** — reduced L/R jitter significantly while preserving traffic avoidance and overtaking. |
| 11 | Combined 8+9+10 | **Worse than 10 alone** — speed changes (8+9) removed too much speed motivation when combined. Agent lost urgency to overtake. |

**Winner: Iter 10 alone** (`steering_penalty: 0.0075 → 0.025`)

**Current best config (iter5 base + iter10):**

| Parameter | Value |
|---|---|
| `smoothing_factor` | 0.5 |
| `vehicles_count` | 12 |
| `npc_speed_scale` | 0.65 |
| `normalize_reward` | false |
| `frame_stack` | 3 |
| `steering_penalty` | 0.025 |

**Remaining problems:**
- **Phantom avoidance:** Sometimes weaves to avoid a car that isn't there, and
  hits a different car during the unnecessary maneuver
- **Blind spots:** Sometimes doesn't see an NPC directly ahead and crashes into it
- **Inconsistency:** Good traffic avoidance most of the time, but occasional failures

**Key insight from Phase 2:** Anti-jitter measures that restrict *capability*
(iters 6-7: change penalty, lateral cap) break traffic avoidance. Anti-jitter
that increases *cost* of unnecessary steering (iter 10: magnitude penalty) works
because the agent can still steer when it needs to — it just doesn't do it
gratuitously.

---

## Isolation Plan — Phase 3 (Iters 12-16)

> **Base config:** Iter 5 + Iter 10 (smoothing=0.5, vehicles=12, npc_speed=0.65,
> normalize=false, frame_stack=3, steering_penalty=0.025)
>
> **Method:** Each iteration changes ONE parameter from this base. NOT cumulative.
> All resume from `checkpoint_00025.pt`.
>
> **Problems to fix:**
> 1. Phantom avoidance — weaves when no car is near (world model hallucination)
> 2. Blind crashes — fails to see car directly ahead (world model gap)
>
> **Root cause analysis:** Both problems point to **world model quality** — the
> agent is making decisions based on inaccurate imagined futures. Phase 3 targets
> model capacity, training intensity, and episode length to improve the world
> model's understanding of traffic dynamics.

### Iter 12 — `imagination_horizon: 20 → 30`

**Targets:** Blind crashes / planning depth  
**Hypothesis:** 20-step horizon may not be far enough to see the consequence of
maintaining speed toward a car ahead. Extending to 30 lets the critic assign
lower value to "fast toward obstacle" states because the imagined crash happens
within the planning window.

**What success looks like:** Fewer head-on crashes; agent brakes or changes lane earlier.  
**What failure looks like:** Training instability or no improvement (horizon not the bottleneck).

### Iter 13 — `max_episode_steps: 75 → 120`

**Targets:** Blind crashes / data quality  
**Hypothesis:** At 75 steps, many episodes truncate before a crash happens —
the agent never sees the consequence of driving straight at traffic. Longer
episodes mean the replay buffer contains more crash events, giving the world
model and critic better signal about what happens when you don't avoid.

**What success looks like:** More diverse replay data; better crash avoidance.  
**What failure looks like:** Episodes just stretch out trailing behavior longer.

### Iter 14 — `world_model_updates_per_cycle: 8 → 16`

**Targets:** World model quality (both problems)  
**Hypothesis:** 8 gradient updates per cycle may be insufficient for the world
model to accurately represent traffic dynamics with frame_stack=3 (3× input
channels). More updates per cycle let the model better learn from each batch
of replay data.

**What success looks like:** Lower reconstruction_mse; more consistent avoidance.  
**What failure looks like:** Overfitting to replay buffer / slower cycles with no gain.

### Iter 15 — `batch_size: 256 → 384`

**Targets:** World model quality (both problems)  
**Hypothesis:** Larger batches give the world model more diverse transitions per
update, reducing gradient noise and improving the consistency of learned dynamics.
Especially important with frame_stack=3 where each sample is 3× larger.

**What success looks like:** Steadier training curves; more reliable avoidance.  
**What failure looks like:** No improvement (noise wasn't the issue) or OOM on GPU.

### Iter 16 — `overtake_reward: 2.5 → 4.0`

**Targets:** Phantom avoidance / reward clarity  
**Hypothesis:** The agent sometimes weaves because it's indifferent between
overtaking and not overtaking — the reward difference isn't large enough to
justify the steering cost (especially with the new steering_penalty=0.025).
Boosting overtake reward makes passing clearly more valuable than staying put,
reducing hesitant half-lane-changes.

**What success looks like:** More committed overtakes; less random weaving.  
**What failure looks like:** Reckless passing → more crashes.

---

## Phase 3 Results — Iters 12-16

All ran 500 cycles from `checkpoint_00025.pt` (iter5 base + iter10).

**Important discovery:** Iter 10 alone **degraded from 500 → 1000 cycles** —
performance peaked around 500 then got worse. This is classic overfitting:
the policy exploits quirks in the world model rather than learning real driving.

| Iter | Knob | Result |
|---|---|---|
| 10 (extended) | steering_penalty=0.025, 1000 cycles | **Degraded past 500** — peaked then worsened. Overfitting to replay buffer. |
| 12 | `imagination_horizon: 20 → 30` | **Third worst** — longer planning into inaccurate world model amplified errors. |
| 13 | `max_episode_steps: 75 → 120` | **Good** — more diverse replay data. Not as strong as iter 10 at 500, but stable. |
| 14 | `world_model_updates: 8 → 16` | **Second worst** — likely overfitting. More updates without more data = memorization. |
| 15 | `batch_size: 256 → 384` | **Good** — steadier training. Not as strong as iter 10 at 500, but stable. |
| 16 | `overtake_reward: 2.5 → 4.0` | **Worst** — reckless passing, more crashes. Boosting reward magnitude backfired. |

**Key insight from Phase 3:** The two best iters (13, 15) both improve **data
quality** without changing what the model learns or how the agent is rewarded.
The three worst iters (16, 14, 12) all tried to change model behavior directly.

- Longer episodes (13) = more transitions going INTO the buffer
- Bigger batches (15) = more diverse sampling coming OUT of the buffer
- Together they attack replay buffer staleness from both sides

**Decision:** Combine 13 + 15 into Iter 17. The goal is to prevent the
overfitting that killed iter 10 past 500 cycles, enabling longer and
better training.

---

## Iter 17 — Combined 13+15

**Base:** iter5 + iter10 (steering_penalty=0.025)  
**Changes:**
- `max_episode_steps: 75 → 120` (from iter 13)
- `batch_size: 256 → 384` (from iter 15)

**Hypothesis:** Iter 10 overfits past 500 cycles because the replay buffer
becomes stale. Longer episodes and bigger batches should provide enough data
diversity to sustain training past 500 and potentially reach a better peak.

**Critical test:** Does performance hold steady or improve past cycle 500
(where iter 10 alone collapsed)?

**Results @ 500:** Completed. Performance comparable to iter 10 at 500.

**Results @ 1000:** **Best at 1000 cycles so far.** Iter 17 is the first
configuration to hold up at 1000 cycles — all previous iters (10, etc.)
degraded significantly past 500. The combined data diversity changes
(longer episodes + bigger batches) are working as hypothesized.

**Results @ 1500:** **Even better than 1000.** Performance continues to
improve — the anti-overfitting effect is holding. Data diversity changes
are providing sustained benefit with no sign of degradation.

**Results @ 2000:** Highest score yet, but **plateaued into passive driving.**
The agent learned to stay in one lane and keep going — it gets most of its
points from simply surviving rather than overtaking. It rarely changes lanes
(~5% of the time), and when it does encounter slower traffic it tends to
crash into the car ahead rather than lane-change around it. No more erratic
wall-smashing or cross-lane crashes, but it’s essentially a “stay-in-lane”
policy that gets lucky. **Plateaued — more cycles won’t help.**

**Status:** Complete. Iter 17 is strong for stability but lacks the traffic
awareness needed for active overtaking.

---

## Iter 18 — Entropy Regularization (code change)

**Base:** iter5 + iter10 (steering_penalty=0.025)  
**Change:** `actor_entropy_weight: 0.0 → 0.003` (new parameter — requires code change)

**What changed in the code (4 files):**
- `config.py`: Added `actor_entropy_weight` field to `TrainingConfig` (default 0.0)
- `actor.py`: Added `distribution()` method returning base `Independent(Normal, 1)`
  for entropy computation (avoids unsupported `TanhTransform.entropy()`)
- `behavior_learning.py`: Actor loss becomes
  `actor_loss = -weighted_mean(returns, weights) - entropy_weight * weighted_mean(entropy, weights)`;
  logs `actor_entropy` metric when weight > 0
- `pipeline.py`: Threads `actor_entropy_weight` from config to `train_behavior_step()`

**Hypothesis:** Actor std collapses after ~500 cycles (Key Discovery #3),
causing the policy to lock in and overfit. An entropy bonus in the actor loss
(borrowed from DreamerV2) encourages the actor to maintain action diversity,
preventing premature std collapse. This is a complementary approach to iter 17's
data-diversity fix — iter 17 addresses buffer staleness, iter 18 addresses
policy collapse directly.

**Why 0.003?** Small enough not to override the return-maximisation objective,
but large enough to measurably resist std collapse. Can be tuned in future iters.

**Results @ 500:** Bad — entropy in isolation (without iter 17's data diversity
changes) does not help. The entropy bonus alone is insufficient to overcome
overfitting when the replay buffer is still stale.

**Results @ 1000:** Very similar to iter 19 @ 1000 — traffic-aware but
aggressive/sporadic. Entropy alone eventually produces similar dynamic
behavior as entropy + data diversity, just slower to get there.

**Results @ 1500:** Not traffic-aware — drives straight into traffic. Less
jittery than iter 19 @ 1500, but overall really bad.

**Results @ 2000:** Same as iter 19 @ 2000 — crashes almost immediately.

**Status:** Complete. Entropy alone is not viable — it produces the same
terminal degradation as the no-entropy configs, just with a different
failure mode (aggressive instead of passive).

---

## Iter 19 — Iter 17 + Entropy

**Base:** iter5 + iter10 + iter17 (steering_penalty=0.025, max_episode_steps=120, batch_size=384)  
**Change:** `actor_entropy_weight: 0.0 → 0.003` (same as iter 18)

**Hypothesis:** Iter 17 was the best at 1000 cycles (data diversity). Iter 18
tests entropy in isolation. This combines both anti-overfitting strategies:
buffer-side diversity (longer episodes + bigger batches from iter 17) and
policy-side exploration pressure (entropy bonus from iter 18). If both
mechanisms are complementary, this should be the strongest configuration.

**Results @ 500:** Good — performance is strong at 500 cycles, on par with
iter 17 at 500. The agent is notably traffic-aware: it actively weaves
around slower vehicles rather than just staying in lane. The combination of
data diversity and entropy regularization is producing the most dynamic
driving behavior seen so far.

**Results @ 1000:** Low score but **qualitatively the best driving behavior.**
The agent is very traffic-aware — it actively weaves around slower vehicles
and makes deliberate lane changes. However, it is too aggressive: tends to
go full throttle too often, which makes it hard to control at high speed.
This leads to oversteering (goes offroad) or clipping cars during lane
changes. The entropy bonus is successfully preventing std collapse and
producing dynamic behavior, but the policy is too exploratory/sporadic.

**Results @ 1500:** Very traffic-aware and good at moving around traffic,
but does it too fast and crashes. Still jittery. Sometimes drives into cars.
Low score because it crashes into walls and cars too often from oversteering.

**Results @ 2000:** Worse than 1500. Crashes almost immediately. Occasionally
dodges 3 cars quickly but then crashes out of bounds. Degraded.

**Status:** Complete. Best qualitative behavior (traffic awareness, active
lane changes) but can't control speed, leading to crashes that tank score.

---

## Phase 4 — Refinement (Iters 20-23)

> **Conclusion from iters 17-19:** All three degrade past their sweet spot.
> The fundamental pattern is that the actor optimizes imagined rewards from
> the world model's reward predictor, which diverges from reality over long
> training. This is an inherent DreamerV1 limitation.
>
> | Iter | @ 2000 | Behavior | Failure Mode |
> |---|---|---|---|
> | 17 (data only) | Highest score | Passive — stays in lane | Crashes into car ahead rather than lane-change |
> | 18 (entropy only) | Crashes immediately | Aggressive | Drives into traffic / off-road |
> | 19 (both) | Crashes immediately | Most traffic-aware | Too fast, can't control at speed |
>
> **Strategy:** Iter 19 has the best *qualitative* behavior (traffic awareness),
> iter 17 has the best *quantitative* score (stability). Phase 4 runs two
> experiments on each base, targeting their specific weakness.
>
> **Method:** All run 500 cycles from `checkpoint_00025.pt`. NOT cumulative.

### Iter 20 — Lower entropy on iter 19 base

**Base:** iter5 + iter10 + iter17 + entropy  
**Change:** `actor_entropy_weight: 0.003 → 0.001`

**Hypothesis:** Iter 19's 0.003 entropy weight made the policy too exploratory,
causing full-throttle aggressive behavior. A gentler 0.001 should preserve the
traffic awareness while letting the policy converge on smoother control.

**Results @ 500:** Goes off-road immediately. Doesn't even reach traffic before
episode ends. Non-functional.

**Results @ 1000:** Still goes off-road immediately. Non-functional at any
cycle count. The iter 19 base with entropy 0.003 produces latent dynamics
that steer off the road regardless of training length.

**Status:** Failed. Off-road at both 500 and 1000.

### Iter 21 — Lower smoothing on iter 19 base

**Base:** iter5 + iter10 + iter17 + entropy (0.003)  
**Change:** `smoothing_factor: 0.5 → 0.35`

**Hypothesis:** Iter 19's agent tries to dodge traffic but can't turn fast
enough at high speed due to action smoothing. Less smoothing = more responsive
steering = better chance of completing the dodge maneuvers the agent is
clearly attempting.

**Results @ 500:** Goes off-road immediately. Same as iter 20 — non-functional.
Less smoothing made the off-road problem worse, not better.

**Results @ 1000:** Still goes off-road immediately. Non-functional.

**Status:** Failed. Off-road at both 500 and 1000.

### Iter 22 — Lower steering penalty on iter 17 base

**Base:** iter5 + iter10 + iter17 (no entropy)  
**Change:** `steering_penalty: 0.025 → 0.015`

**Hypothesis:** Iter 17 @ 2000 became passive because `steering_penalty=0.025`
makes lane changes too costly relative to the overtake reward. The original
0.0075 was too low (jitter), 0.025 is too high (won't steer at all after
extended training). 0.015 is a middle ground — still higher than baseline but
low enough that an overtake reward can justify a lane change.

**Results @ 500:** Goes off-road immediately. Despite iter 17 base never having
off-road issues at 0.025, reducing to 0.015 allows too much steering freedom
and the agent steers off the road.

**Results @ 1000:** Still goes off-road immediately. Non-functional.

**Status:** Failed. Off-road at both 500 and 1000.

### Iter 23 — Lower steering penalty + gentle entropy on iter 17 base

**Base:** iter5 + iter10 + iter17 (no entropy)  
**Changes:**
- `steering_penalty: 0.025 → 0.015`
- `actor_entropy_weight: 0.0 → 0.001`

**Hypothesis:** Combines iter 22's cheaper lane changes with a gentle entropy
nudge (lower than iter 19's too-aggressive 0.003). The steering fix makes lane
changes affordable; the entropy bonus ensures the actor actually tries them
instead of collapsing to a stay-in-lane policy.

**Results @ 500:** Doesn't go fast enough to score. When it does move, it
crashes. Trails behind traffic and stays back. The entropy regularization
prevents the off-road issue seen in iter 22 (same steering penalty without
entropy), but makes the agent too conservative — similar to iter 17's passive
behavior but worse because it can't even maintain speed.

**Results @ 1000:** Scores low but is traffic-aware and actively dodges cars.
The gentle entropy bonus (0.001) keeps the actor's std from collapsing, and
with more training the policy learns to time lane changes around oncoming
traffic. This is the only Phase 4 configuration that produced useful behavior.
Critically, the entropy bonus (borrowed from DreamerV2's actor loss design)
prevents the off-road failure that killed iter 22 at the same steering penalty.

**Status:** SELECTED as final behaviour-optimized run alongside iter 17.

### Iter 24 — Mid steering penalty on iter 17 base

**Base:** iter5 + iter10 + iter17 (no entropy)  
**Change:** `steering_penalty: 0.025 → 0.020`

**Hypothesis:** Iter 22 (0.015) went off-road — too much steering freedom.
Iter 17 (0.025) is passive — won't lane-change at all past ~1500 cycles.
0.020 is the tightest middle ground: still penalizes jitter but may allow
occasional lane changes when the overtake reward is high enough.

**Results @ 500:** Goes off-road immediately, same as iters 20-22.

**Results @ 1000:** Still goes off-road immediately. The 0.020 steering
penalty is not enough constraint — without entropy regularization, the actor's
std collapses to a degenerate policy that steers off the road.

**Status:** Failed. Off-road at both 500 and 1000.

---

## Phase 4 Summary

| Iter | Base | Change | @500 | @1000 | Status |
|------|------|--------|------|-------|--------|
| 20 | iter19 | entropy 0.003→0.001 | Off-road | Off-road | Failed |
| 21 | iter19 | smoothing 0.5→0.35 | Off-road | Off-road | Failed |
| 22 | iter17 | steering 0.025→0.015 | Off-road | Off-road | Failed |
| 23 | iter17 | steering 0.015 + entropy 0.001 | Passive | **Traffic-aware** | **Selected** |
| 24 | iter17 | steering 0.025→0.020 | Off-road | Off-road | Failed |

**Key insight:** Reducing steering penalty alone (iters 22, 24) causes off-road
failure. Adding entropy regularization (iter 23) prevents this by maintaining
the actor's exploration distribution, keeping it from collapsing to a degenerate
steering policy. This is the DreamerV2 insight applied to DreamerV1: the entropy
bonus serves as a stabilizer, not just an exploration aid.

---

## Key Discoveries

### 1. `actor_init_std: 5.0` causes tanh saturation (bang-bang)

- `std = softplus(raw_std + init_std) + min_std`
- `softplus(5.0) ≈ 5.0` → noise so large that tanh squashes everything to ±1
- Agent can only do hard-left/hard-right/full-gas/full-brake
- **Fix:** `actor_init_std: 1.0` → `softplus(1.0) ≈ 1.31` → smooth exploration

### 2. `offroad_terminal: false` enables the spinning exploit

- With continuous control, the agent can steer off-road and spin indefinitely
- Combined with the +0.4 existence baseline from normalize_reward, spinning is a
  "safe" local optimum
- **Rule:** Always keep `offroad_terminal: true` with continuous control

### 3. No entropy bonus in actor loss

- The actor loss is purely `-weighted_mean(returns, weights)` — no entropy term
- Once the actor's learned std collapses (which happens fast when trailing is
  safest), there's nothing forcing it to explore lane changes
- This creates a feedback loop: trailing data → world model only knows trailing
  → imagined overtakes look bad → policy stays trailing

### 4. Policy overfitting past ~500 cycles

- Iter 10 peaked at ~500 cycles then degraded when extended to 1000
- The world model memorizes replay buffer quirks; the actor exploits them
- **Fix:** Increase data diversity — longer episodes (more transitions in) and
  bigger batches (more diverse sampling out) prevent staleness
- Don't assume "more training = better" — always check the 500→1000 trend

### 5. Actor optimizes phantom imagined rewards (DreamerV1 limitation)

- The actor **never sees real rewards** — it maximizes returns from the world
  model's reward predictor applied to imagined latent states
- During behavior learning the world model is frozen — the actor optimizes a
  static fantasy that becomes increasingly disconnected from reality
- This explains all failure modes at high cycle counts:
  - **Passive (iter 17):** World model learned "stay in lane" is reliably
    rewarding in imagination; actor exploits this safe local optimum
  - **Aggressive (iters 18/19):** Entropy keeps actor exploring, and it finds
    that "go fast + dodge" looks amazing in imagination, but the world model's
    dynamics are wrong at high speed — imagined dodges succeed, real ones crash
- **Implication:** There is a sweet spot for each config. More training past it
  means deeper exploitation of imagined reward inaccuracies

---

## Parameters NOT to Change

These were evaluated and deliberately rejected:

| Suggestion | Why Rejected |
|---|---|
| **Trailing penalty** (negative reward for being behind NPCs) | "Kamikaze Trap" — punishes the agent for being near traffic at all, encourages suicidal passes |
| **Non-terminal collisions** (`collision_terminal: false`) | Creates "zombie states" — agent keeps collecting penalties after crashing, poisons the replay buffer with meaningless transitions |

---
