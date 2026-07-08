"""Shared feature-vector construction for Phase 3 RL.

This is the single place the 26-dim observation is built from a SensorState,
used by both training/rl/torcs_gym_env.py and drivers/rl/driver.py. Keeping it
in one function (instead of duplicating it, as the previous RL attempt did)
guards against the exact bug that sank that attempt: the RL observation space
silently drifting out of sync with what the BC network was actually trained on
(see git history around commits 727593b / d357744 / 074c1ee for the incident —
BC used [speed, trackPos, angle, rpm, gear, track[6], track[12], track[18]]
while RL used different track indices, causing a persistent zero-steering bug
that was never fully resolved before the whole Phase 3 attempt was removed).

Layout matches _DRIVER/driver.py's sensor_vec exactly:
    [angle, speed, speedY, speedZ, trackPos, *track(19), rpm, gear]
"""

from __future__ import annotations

import numpy as np

from torcs_env.sensors import SensorState

FEATURE_DIM = 26


def build_feature_vector(state: SensorState) -> np.ndarray:
    return np.array(
        [
            state.angle,
            state.speed,
            state.speedY,
            state.speedZ,
            state.trackPos,
            *state.track,
            state.rpm,
            float(state.gear),
        ],
        dtype=np.float32,
    )
