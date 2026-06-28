"""Reward function for TORCS RL training.

Tune the weights here to shape the agent's behaviour without changing gym_env.py.
"""

from __future__ import annotations

import math

from torcs_env.sensors import SensorState

# --- Weights (easy to experiment with) ---
PROGRESS_WEIGHT: float = 1.0     # forward velocity (m/s)
DEVIATION_WEIGHT: float = 1.0    # lateral-drift penalty
DAMAGE_WEIGHT: float = 100.0     # penalty per unit of new damage
OFF_TRACK_PENALTY: float = 2.0   # flat per-step penalty when |trackPos| > 1
STEER_SMOOTH_WEIGHT: float = 0.05  # penalise jerky steering
LAP_BONUS: float = 500.0         # one-shot bonus on lap completion


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

    # Positive: forward velocity projected onto the track axis
    progress = speed_ms * math.cos(curr.angle) * PROGRESS_WEIGHT

    # Negative: speed × lateral deviation encourages staying centred
    deviation = abs(speed_ms) * abs(curr.trackPos) * DEVIATION_WEIGHT

    # Negative: crash/damage penalty
    damage_delta = max(0.0, curr.damage - prev.damage)
    damage_penalty = damage_delta * DAMAGE_WEIGHT

    # Negative: per-step off-track penalty
    off_track = OFF_TRACK_PENALTY if abs(curr.trackPos) > 1.0 else 0.0

    # Negative: steering jerk penalty (smooth driving)
    steer_jerk = (curr_steer - prev_steer) ** 2 * STEER_SMOOTH_WEIGHT

    # Positive one-time bonus for completing a lap
    bonus = LAP_BONUS if lap_completed else 0.0

    return progress - deviation - damage_penalty - off_track - steer_jerk + bonus
