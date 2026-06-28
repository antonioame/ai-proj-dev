"""Reward function for TORCS RL training.

Tune the weights here to shape the agent's behaviour without changing gym_env.py.
"""

from __future__ import annotations

import math

from torcs_env.sensors import SensorState

# --- Weights (easy to experiment with) ---
# IMPROVED v1: Prioritize steering & lane-keeping over raw speed
PROGRESS_WEIGHT: float = 0.5     # forward velocity (REDUCED: encourage efficiency)
DEVIATION_WEIGHT: float = 5.0    # lateral-drift penalty (INCREASED 5×)
DAMAGE_WEIGHT: float = 100.0     # penalty per unit of new damage
OFF_TRACK_PENALTY: float = 20.0  # flat per-step penalty when |trackPos| > 1 (INCREASED 10×)
STEER_SMOOTH_WEIGHT: float = 0.1  # penalise jerky steering (slight increase)
LAP_BONUS: float = 500.0         # one-shot bonus on lap completion
LANE_CENTER_BONUS: float = 1.0   # reward for staying centered (NEW)


def compute_reward(
    prev: SensorState,
    curr: SensorState,
    prev_steer: float = 0.0,
    curr_steer: float = 0.0,
    lap_completed: bool = False,
) -> float:
    """Return scalar reward for one simulation step.

    Parameters
    ----------
    prev, curr:
        Sensor states before and after the action was applied.
    prev_steer, curr_steer:
        Steering values at the previous and current step (for smoothness term).
    lap_completed:
        True when the car just crossed the start/finish line.
    """
    speed_ms = curr.speed / 3.6  # km/h → m/s

    # Positive: forward velocity projected onto the track axis (REDUCED to not dominate)
    progress = speed_ms * math.cos(curr.angle) * PROGRESS_WEIGHT

    # Negative: speed × lateral deviation STRONGLY encourages staying centred
    deviation = abs(speed_ms) * abs(curr.trackPos) * DEVIATION_WEIGHT

    # Positive: reward for staying centered (trackPos near 0)
    lane_center = (1.0 - abs(curr.trackPos)) * LANE_CENTER_BONUS if abs(curr.trackPos) < 1.0 else 0.0

    # Negative: crash/damage penalty
    damage_delta = max(0.0, curr.damage - prev.damage)
    damage_penalty = damage_delta * DAMAGE_WEIGHT

    # Negative: per-step off-track penalty (MUCH STRONGER - triggers EARLY)
    # Penalize progressively: 0.5 → 5pts, 0.75 → 10pts, 1.0+ → 20pts
    if abs(curr.trackPos) > 1.0:
        off_track = OFF_TRACK_PENALTY  # 20.0
    elif abs(curr.trackPos) > 0.75:
        off_track = 10.0  # Strong warning when approaching edge
    elif abs(curr.trackPos) > 0.5:
        off_track = 5.0   # Gentle nudge to stay centered
    else:
        off_track = 0.0

    # Negative: steering jerk penalty (smooth driving)
    steer_jerk = (curr_steer - prev_steer) ** 2 * STEER_SMOOTH_WEIGHT

    # Positive one-time bonus for completing a lap
    bonus = LAP_BONUS if lap_completed else 0.0

    return progress - deviation - damage_penalty - off_track - steer_jerk + lane_center + bonus
