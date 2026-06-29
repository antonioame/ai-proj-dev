"""OptimalLineDriver: position-indexed trajectory follower with late hard braking.

Advantages over rule_based:
  - Late braking: goes flat-out until the last mathematically safe moment.
  - Earlier throttle: knows the track ahead, applies power on corner exit.
  - ABS: prevents wheel lockup at BRAKE_MAX=1.0.
  - TCS: prevents wheelspin on acceleration.

Requires: torcs_env/track_data/track_map.json
Build map: python scripts/build_track_map.py --telemetry data/<file>.csv
"""
from __future__ import annotations

import math
from pathlib import Path

from drivers.base_driver import BaseDriver
from drivers.optimal.trajectory import Trajectory, BACKWARD_DECEL_FACTOR
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from torcs_env.track_map import TrackMap

_DEFAULT_MAP_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "torcs_env" / "track_data" / "track_map.json"
)

# --------------- Steering ---------------
STEER_ANGLE_GAIN:   float = 1.2    # angle correction (was 1.6 — too twitchy)
STEER_LINE_GAIN:    float = 0.25   # bias toward racing line (was 0.40 — too aggressive)
STEER_LOCK:         float = 0.785398
STEER_SMOOTH_SPEED: float = 75.0   # apply EMA smoothing below this speed (was 50)
STEER_SMOOTH_ALPHA: float = 0.25   # EMA weight for new steer (was 0.35 — more damping)
TARGET_LINE_SCALE:  float = 0.50   # blend trajectory line with centre (was 0.70)

# --------------- Speed control ---------------
BRAKE_MAX:        float = 1.0
SCAN_AHEAD_M:     float = 200.0   # look-ahead for braking (was 300 — too far ahead)
BRAKE_MARGIN_M:   float = 40.0    # extra buffer on top of braking distance (was 85)
THROTTLE_BASE:    float = 0.35    # minimum throttle in slow corners

# ABS
WHEEL_RADIUS:        float = 0.33
ABS_SLIP_THRESHOLD:  float = 0.80

# TCS (traction control)
TCS_SLIP_THRESHOLD: float = 1.25  # rear/expected > this → cut throttle
TCS_SLIP_GAIN:      float = 3.0
TCS_MIN_ACCEL:      float = 0.20

# --------------- Gear ---------------
RPM_UPSHIFT:   float = 9000.0
RPM_DOWNSHIFT: dict[int, float] = {6: 6800, 5: 6300, 4: 5800, 3: 4300, 2: 3500}
RPM_DOWNSHIFT_DEFAULT: float = 3000.0
GEAR_SPEED_CAPS: list[tuple[float, int]] = [(15.0, 1), (45.0, 2), (75.0, 3)]

# --------------- Safety / startup ---------------
STARTUP_STEPS:     int   = 200    # conservative start phase (was 120)
FALLBACK_TRACKPOS: float = 1.0    # abs(trackPos) > this → recovery (was 1.2)


class OptimalLineDriver(BaseDriver):
    """Track-position-indexed trajectory controller with late hard braking."""

    def __init__(self, map_path: Path | None = None) -> None:
        self._map_path = map_path or _DEFAULT_MAP_PATH
        self._trajectory: Trajectory | None = None
        self._step_count: int = 0
        self._prev_steer: float = 0.0

    def _load_trajectory(self) -> None:
        track_map = TrackMap.load(self._map_path)
        self._trajectory = Trajectory.from_track_map(track_map)

    def reset(self) -> None:
        self._step_count = 0
        self._prev_steer = 0.0

    def on_restart(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    def step(self, state: SensorState) -> Action:
        if self._trajectory is None:
            self._load_trajectory()

        self._step_count += 1

        # startup: conservative steer (0.4×), full throttle, speed-capped gear
        if self._step_count <= STARTUP_STEPS:
            steer = self._steer(state, 0.0) * 0.4
            self._prev_steer = steer
            return Action(steer=steer, accel=1.0, brake=0.0,
                          gear=self._startup_gear(state)).clamp()

        # off-track: recovery
        if abs(state.trackPos) > FALLBACK_TRACKPOS:
            return self._recovery(state)

        assert self._trajectory is not None
        traj = self._trajectory
        v_cur = state.speed

        # scan ahead for the tightest corner in SCAN_AHEAD_M
        v_min, d_to_min = traj.min_speed_ahead(state.distFromStart, SCAN_AHEAD_M)

        decel_max = BRAKE_MAX * BACKWARD_DECEL_FACTOR  # (km/h)²/m
        accel: float = 0.0
        brake: float = 0.0

        if v_min >= v_cur - 2.0:
            accel = 1.0
        else:
            d_onset = (v_cur ** 2 - v_min ** 2) / (2.0 * decel_max)
            if d_to_min > d_onset + BRAKE_MARGIN_M:
                accel = 1.0
            else:
                d_safe = max(1.0, d_to_min)
                required = (v_cur ** 2 - v_min ** 2) / (2.0 * d_safe)
                bp = min(BRAKE_MAX, required / BACKWARD_DECEL_FACTOR)
                brake = self._apply_abs(bp, state)

        # slow corner: ensure we don't coast
        if brake == 0.0 and v_cur < 80.0:
            target_here = traj.lookup_speed(state.distFromStart)
            if v_cur < target_here:
                accel = max(accel, min(1.0, THROTTLE_BASE + (target_here - v_cur) * 0.02))

        # traction control: cut accel if rear wheels spin
        if accel > 0.0:
            accel = self._apply_tcs(accel, state)

        target_tp = traj.lookup_line(state.distFromStart) * TARGET_LINE_SCALE
        steer = self._steer(state, target_tp)
        gear = self._compute_gear(state, braking=(brake > 0))

        return Action(steer=steer, accel=accel, brake=brake, gear=gear).clamp()

    # ------------------------------------------------------------------
    def _steer(self, state: SensorState, target_tp: float) -> float:
        line_err = state.trackPos - target_tp
        raw = state.angle * STEER_ANGLE_GAIN - line_err * STEER_LINE_GAIN
        steer = max(-0.85, min(0.85, raw / STEER_LOCK))
        if state.speed < STEER_SMOOTH_SPEED:
            steer = (
                self._prev_steer * (1.0 - STEER_SMOOTH_ALPHA)
                + steer * STEER_SMOOTH_ALPHA
            )
        self._prev_steer = steer
        return steer

    def _apply_abs(self, brake: float, state: SensorState) -> float:
        if brake < 0.05 or state.speed < 10.0 or len(state.wheelSpinVel) < 2:
            return brake
        speed_ms = state.speed / 3.6
        expected = speed_ms / WHEEL_RADIUS
        if expected < 1.0:
            return brake
        front = (state.wheelSpinVel[0] + state.wheelSpinVel[1]) / 2.0
        ratio = front / expected
        if ratio < ABS_SLIP_THRESHOLD:
            lockup = ABS_SLIP_THRESHOLD - ratio
            brake = max(0.0, brake * (1.0 - lockup / ABS_SLIP_THRESHOLD))
        return brake

    def _apply_tcs(self, accel: float, state: SensorState) -> float:
        if state.speed < 5.0 or len(state.wheelSpinVel) < 4:
            return accel
        speed_ms = state.speed / 3.6
        expected = speed_ms / WHEEL_RADIUS
        if expected < 1.0:
            return accel
        rear_avg = (state.wheelSpinVel[2] + state.wheelSpinVel[3]) / 2.0
        slip = rear_avg / expected - TCS_SLIP_THRESHOLD
        if slip > 0.0:
            accel = min(accel, max(TCS_MIN_ACCEL, 1.0 - slip * TCS_SLIP_GAIN))
        return accel

    def _startup_gear(self, state: SensorState) -> int:
        if state.speed < 15.0:
            return 1
        if state.speed < 45.0:
            return 2
        return 3

    def _compute_gear(self, state: SensorState, braking: bool = False) -> int:
        g = max(1, state.gear)
        if state.rpm > RPM_UPSHIFT and g < 6:
            g += 1
        else:
            margin = 800.0 if braking else 0.0
            thresh = RPM_DOWNSHIFT.get(g, RPM_DOWNSHIFT_DEFAULT)
            if state.rpm < (thresh - margin) and g > 1:
                g -= 1
        for v_cap, g_cap in GEAR_SPEED_CAPS:
            if state.speed < v_cap:
                g = min(g, g_cap)
                break
        return g

    def _recovery(self, state: SensorState) -> Action:
        steer = -math.copysign(0.5, state.trackPos)
        # light brake to kill speed, then steer back
        brake = 0.2 if state.speed > 30.0 else 0.0
        accel = 0.0 if brake > 0 else 0.3
        return Action(steer=steer, accel=accel, brake=brake,
                      gear=max(1, state.gear)).clamp()
