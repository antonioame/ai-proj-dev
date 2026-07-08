"""Versioned reward functions for Phase 3 RL (REINFORCEMENT_LEARNING.md Section 4).

Every variant is a self-contained RewardVersion so a training run can log
exactly which formula produced its results (Section 4.2: "log the reward
formula version used for every training run").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from torcs_env.sensors import SensorState

StepRewardFn = Callable[[SensorState, SensorState], float]


@dataclass(frozen=True)
class RewardVersion:
    name: str
    off_track_penalty: float
    lap_bonus: float
    step_fn: StepRewardFn


def _baseline_step(prev_state: SensorState, state: SensorState) -> float:
    """Section 4.1 baseline: r = v*cos(angle) - v*|sin(angle)| - v*|trackPos|."""
    v = state.speed
    return v * math.cos(state.angle) - v * abs(math.sin(state.angle)) - v * abs(state.trackPos)


# Weight keeps the progress term in the same rough magnitude as the speed
# terms above (Section 4.2.5, "normalization") — distRaced deltas per 50 Hz
# step are on the order of a few metres at racing speed, same order as v*cos.
_PROGRESS_WEIGHT = 0.5
_STANDING_STILL_KMH = 5.0
_STANDING_STILL_PENALTY = -1.0


def _refined_step(prev_state: SensorState, state: SensorState) -> float:
    """Section 4.2 refinements over baseline_v1:

    1. Progress reward — proportional to distRaced covered this step, not
       just instantaneous speed. Targets the actual success metric (a
       completed, fast lap) and is the course's own mitigation for reward
       hacking (Section 4.3).
    2. Standing-still guard — flat penalty below _STANDING_STILL_KMH so
       idling/spinning-in-place can't out-score careful driving.

    Off-track termination severity (refinement 3) lives in
    RewardVersion.off_track_penalty, applied by the caller on termination.
    """
    base = _baseline_step(prev_state, state)
    progress = _PROGRESS_WEIGHT * max(0.0, state.distRaced - prev_state.distRaced)
    standing_penalty = _STANDING_STILL_PENALTY if state.speed < _STANDING_STILL_KMH else 0.0
    return base + progress + standing_penalty


REWARD_VERSIONS: dict[str, RewardVersion] = {
    "baseline_v1": RewardVersion(
        name="baseline_v1",
        off_track_penalty=-100.0,
        lap_bonus=50.0,
        step_fn=_baseline_step,
    ),
    "refined_v2": RewardVersion(
        name="refined_v2",
        # Tuned higher than baseline per Section 4.2.3: the project's hard
        # constraint is "no crashes, minimal off-track excursions", so the
        # policy should strongly avoid track exits rather than trade a small
        # chance of exiting for marginally higher speed.
        off_track_penalty=-200.0,
        lap_bonus=100.0,
        step_fn=_refined_step,
    ),
}
