"""Environment factory and reward-shaping wrappers for Highway-Env.

This module constructs a Gymnasium-compatible highway-env environment
tailored for the Tiny Dreamer agent.  Key responsibilities:

* **Observation configuration** — sets up grayscale frame-stacked pixel
  observations suitable for CNN encoding.
* **Reward shaping** — applies configurable penalty/bonus wrappers on top
  of the base highway-env reward to encourage smoother driving and
  reward overtaking (``DrivingPenaltyRewardWrapper``).
* **NPC speed adjustment** — optionally slows down traffic vehicles to
  make the learning problem easier early in training
  (``NPCSpeedAdjustmentWrapper``).
* **Wrapper ordering** — ensures that Highway-Env’s lazy observation
  construction is completed (via an initial ``reset()``) *before*
  wrappers are applied.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from tiny_dreamer_highway.config import EnvConfig


class DrivingPenaltyRewardWrapper(gym.Wrapper):
    """Adds penalty and bonus terms to the base highway-env reward.

    Shaping terms (all configurable, all zero by default):

    * **Steering penalty** — penalizes large lateral actions to
      discourage excessive swerving.
    * **Steering change penalty** — penalizes abrupt changes in
      lateral action (jerk reduction).
    * **Off-road penalty** — flat penalty when the ego vehicle leaves
      the road surface.
    * **Overtake bonus** — rewards the ego for passing NPC vehicles.
      Each NPC can only trigger one bonus per episode (tracked by
      ``_rewarded_overtakes``).

    When highway-env’s ``normalize_reward`` is enabled, the base reward
    is mapped to [0, 1].  This wrapper scales its additive terms by the
    same normalization factor so that config values maintain proportional
    meaning.
    """

    def __init__(self, env: gym.Env, config: EnvConfig) -> None:
        super().__init__(env)
        self._config = config
        self._previous_lateral_action = 0.0
        # Maps vehicle id() → (object reference, x-position) for vehicles
        # that were ahead of ego at the start of the step
        self._previous_ahead_vehicles: dict[int, tuple[Any, float]] = {}
        # Set of id()s already rewarded to prevent double-counting
        self._rewarded_overtakes: set[int] = set()

        # When normalize_reward is True, highway-env maps the raw reward from
        # [collision_reward, high_speed_reward + right_lane_reward] to [0, 1].
        # Scale additive shaping terms by the same factor so config values
        # keep proportional meaning regardless of normalisation.
        rc = config.reward
        if rc.normalize_reward:
            raw_span = (rc.high_speed_reward + rc.right_lane_reward) - rc.collision_reward
            self._additive_scale = 1.0 / max(raw_span, 1e-8)
        else:
            self._additive_scale = 1.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._previous_lateral_action = 0.0
        self._rewarded_overtakes.clear()
        observation = self.env.reset(seed=seed, options=options)
        ego_x = self._get_ego_x()
        all_vehicles = self._snapshot_all_vehicles()
        self._previous_ahead_vehicles = (
            {vid: info for vid, info in all_vehicles.items() if info[1] > ego_x}
            if ego_x is not None
            else {}
        )
        return observation

    def step(self, action: Any):
        """Step the environment and apply reward shaping."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Apply driving-style penalties and overtake bonus, scaled to
        # match the base reward’s normalization range.
        penalty = self._compute_penalty(action) * self._additive_scale
        bonus = self._compute_overtake_bonus() * self._additive_scale
        shaped_reward = float(reward) - penalty + bonus
        return observation, shaped_reward, terminated, truncated, info

    def _compute_penalty(self, action: Any) -> float:
        """Compute steering and off-road penalties for this step."""
        penalties = self._config.reward
        lateral_action = _extract_lateral_action(action, self._config)

        # Penalize the magnitude of lateral steering
        penalty = penalties.steering_penalty * abs(lateral_action)
        # Penalize abrupt changes (steering jerk)
        penalty += penalties.steering_change_penalty * abs(
            lateral_action - self._previous_lateral_action
        )
        self._previous_lateral_action = lateral_action

        # Flat penalty for driving off the road surface
        vehicle = getattr(self.env.unwrapped, "vehicle", None)
        if vehicle is not None and not bool(getattr(vehicle, "on_road", True)):
            penalty += penalties.offroad_penalty
        return float(penalty)

    def _compute_overtake_bonus(self) -> float:
        """Award bonus for each NPC vehicle the ego has passed this step.

        An overtake is counted only when a vehicle:
        1. Was previously ahead of ego (recorded in ``_previous_ahead_vehicles``).
        2. Still exists on the road (not despawned between steps).
        3. Has the same Python object identity (``id()`` not recycled).
        4. Is now behind the ego vehicle.
        Each NPC can only trigger one bonus per episode.
        """
        overtake_reward = self._config.reward.overtake_reward
        if overtake_reward <= 0.0:
            self._previous_ahead_vehicles = {}
            return 0.0

        ego_x = self._get_ego_x()
        if ego_x is None:
            self._previous_ahead_vehicles = {}
            return 0.0

        all_current = self._snapshot_all_vehicles()
        current_ahead = {
            vid: info for vid, info in all_current.items() if info[1] > ego_x
        }

        # A vehicle counts as truly overtaken only when it:
        # 1. was previously ahead of ego,
        # 2. still exists on the road (not despawned),
        # 3. has the same object identity (Python id() not recycled), and
        # 4. is now behind the ego vehicle.
        bonus_count = 0
        for vid, (prev_ref, _) in self._previous_ahead_vehicles.items():
            if vid in self._rewarded_overtakes:
                continue
            if vid not in all_current:
                continue  # vehicle despawned — not a real overtake
            curr_ref, curr_x = all_current[vid]
            if curr_ref is not prev_ref:
                continue  # Python recycled the id for a different object
            if curr_x < ego_x:
                bonus_count += 1
                self._rewarded_overtakes.add(vid)

        self._previous_ahead_vehicles = current_ahead
        return float(overtake_reward * bonus_count)

    def _get_ego_x(self) -> float | None:
        ego = getattr(self.env.unwrapped, "vehicle", None)
        if ego is None:
            return None
        pos = np.asarray(getattr(ego, "position", []), dtype=np.float32).reshape(-1)
        return float(pos[0]) if pos.size >= 1 else None

    def _snapshot_all_vehicles(self) -> dict[int, tuple[Any, float]]:
        """Return ``{id(v): (v, x)}`` for every non-ego vehicle on the road."""
        ego = getattr(self.env.unwrapped, "vehicle", None)
        road = getattr(self.env.unwrapped, "road", None)
        if road is None:
            return {}
        result: dict[int, tuple[Any, float]] = {}
        for v in getattr(road, "vehicles", []):
            if v is ego:
                continue
            pos = np.asarray(getattr(v, "position", []), dtype=np.float32).reshape(-1)
            if pos.size >= 1:
                result[id(v)] = (v, float(pos[0]))
        return result


class NPCSpeedAdjustmentWrapper(gym.Wrapper):
    """Scale NPC vehicle speeds on reset to adjust traffic difficulty.

    When ``npc_speed_scale < 1.0``, every non-ego vehicle’s speed and
    target speed are multiplied by this factor immediately after each
    ``reset()``.  This makes the driving task easier by giving the ego
    more time to react.
    """

    def __init__(self, env: gym.Env, config: EnvConfig) -> None:
        super().__init__(env)
        self._npc_speed_scale = config.npc_speed_scale

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation = self.env.reset(seed=seed, options=options)
        self._apply_npc_speed_scale()
        return observation

    def _apply_npc_speed_scale(self) -> None:
        if self._npc_speed_scale >= 1.0:
            return

        unwrapped = self.env.unwrapped
        ego = getattr(unwrapped, "vehicle", None)
        road = getattr(unwrapped, "road", None)
        if road is None:
            return

        for vehicle in getattr(road, "vehicles", []):
            if vehicle is ego:
                continue
            if hasattr(vehicle, "speed"):
                vehicle.speed = float(vehicle.speed) * self._npc_speed_scale
            if hasattr(vehicle, "target_speed") and getattr(vehicle, "target_speed") is not None:
                vehicle.target_speed = float(vehicle.target_speed) * self._npc_speed_scale


def _extract_lateral_action(action: Any, config: EnvConfig) -> float:
    """Extract the lateral (steering) component from a raw action.

    Returns 0.0 for discrete action spaces (no continuous lateral dim).
    For continuous actions, returns the lateral component based on
    whether longitudinal control is also enabled (determines index).
    """
    if config.action.is_discrete:
        return 0.0
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.size == 0 or not config.action.lateral:
        return 0.0
    if config.action.longitudinal and action_array.size >= 2:
        return float(action_array[1])
    return float(action_array[0])


def _should_apply_reward_wrapper(config: EnvConfig) -> bool:
    """Check whether any reward-shaping terms are active.

    The ``DrivingPenaltyRewardWrapper`` is only applied when at least
    one penalty or bonus is non-zero.  Discrete action spaces skip the
    wrapper entirely since steering penalties are not meaningful.
    """
    if config.action.is_discrete:
        return False
    reward_config = config.reward
    return (
        reward_config.overtake_reward > 0.0
        or reward_config.offroad_penalty > 0.0
        or reward_config.steering_penalty > 0.0
        or reward_config.steering_change_penalty > 0.0
    )


def build_highway_env_kwargs(config: EnvConfig) -> dict[str, Any]:
    """Build the configuration dict passed to ``env.unwrapped.configure()``.

    Translates the project-specific ``EnvConfig`` into the keyword
    arguments that highway-env expects for observation, action, road,
    and reward configuration.
    """
    if config.action.is_discrete:
        action_block: dict[str, Any] = {
            "type": "DiscreteMetaAction",
            "target_speeds": list(
                np.linspace(
                    config.reward.reward_speed_range[0],
                    config.reward.reward_speed_range[1],
                    num=config.action.num_actions,
                )
            ),
        }
    else:
        action_block = {
            "type": "ContinuousAction",
            "longitudinal": config.action.longitudinal,
            "lateral": config.action.lateral,
        }
    return {
        "observation": {
            "type": "GrayscaleObservation",
            # upstream GrayscaleObservation interprets this as (width, height)
            "observation_shape": (config.observation_width, config.observation_height),
            "stack_size": config.frame_stack,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        },
        "action": action_block,
        "lanes_count": config.lanes_count,
        "vehicles_count": config.vehicles_count,
        "simulation_frequency": config.simulation_frequency,
        "policy_frequency": config.policy_frequency,
        # upstream duration is in seconds; convert agent steps to seconds
        "duration": config.max_episode_steps / config.policy_frequency,
        "collision_reward": config.reward.collision_reward,
        "right_lane_reward": config.reward.right_lane_reward,
        "high_speed_reward": config.reward.high_speed_reward,
        "lane_change_reward": config.reward.lane_change_reward,
        "reward_speed_range": list(config.reward.reward_speed_range),
        "normalize_reward": config.reward.normalize_reward,
        "offroad_terminal": config.reward.offroad_terminal,
    }


def make_highway_env(config: EnvConfig):
    """Create and configure a highway-env Gymnasium environment.

    Construction order matters:

    1. ``gym.make()`` creates the bare environment.
    2. ``configure()`` applies grayscale observation and action settings.
    3. ``env.reset()`` **must** be called before wrapping because
       Highway-Env lazily builds its ``GrayscaleObservation`` handler
       (and updates ``observation_space``) only on the first reset.
       Without this, wrappers would see the wrong observation space.
    4. Optional wrappers (NPC speed, reward shaping) are applied last.
    """
    import highway_env  # noqa: F401  — registers the highway-v0 env ID

    env = gym.make(config.env_id, render_mode="rgb_array")
    env.unwrapped.configure(build_highway_env_kwargs(config))
    
    # ---------------
    # Force Highway-Env to physically build the Grayscale observation 
    # objects and update its observation_space BEFORE we wrap it.
    env.reset()
    # ---------------
    
    if config.npc_speed_scale < 1.0:
        env = NPCSpeedAdjustmentWrapper(env, config)
    if _should_apply_reward_wrapper(config):
        env = DrivingPenaltyRewardWrapper(env, config)
    return env
