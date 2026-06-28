"""OptimalLineDriver: position-indexed trajectory follower with late hard braking.

Key improvements over the reactive rule_based driver:
  - Late braking: goes flat-out until exactly the last safe moment,
    then brakes hard. The reactive driver brakes when a proximity sensor
    triggers; this driver brakes when mathematics require it.
  - Earlier throttle: knows the track ahead so applies power at corner exit.
  - ABS: prevents wheel lockup, allowing BRAKE_MAX=0.90 safely.

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
STEER_ANGLE_GAIN:   float = 1.3     # Reduced to 1.3 - let braking dominate in curves, less steering fight
STEER_LINE_GAIN:    float = 0.20    # trackPos error → steer correction
STEER_LOCK:         float = 0.785398
STEER_SMOOTH_SPEED: float = 50.0    # Adjust for curve sensitivity
STEER_SMOOTH_ALPHA: float = 0.35

# --------------- Speed control ---------------
BRAKE_MAX:        float = 0.95      # Increased from 0.90 for harder braking
SCAN_AHEAD_M:     float = 220.0     # Restored to see curves early (was 150, now 220)
BRAKE_MARGIN_M:   float = 32.0      # Increased to 32 - brake well before sharp turns
THROTTLE_BASE:    float = 0.70      # minimum throttle when below target speed

# ABS
WHEEL_RADIUS:        float = 0.33
ABS_SLIP_THRESHOLD:  float = 0.80

# --------------- Gear ---------------
RPM_UPSHIFT:   float = 9000.0
RPM_DOWNSHIFT: dict[int, float] = {6: 6800, 5: 6300, 4: 5800, 3: 4300, 2: 3500}
RPM_DOWNSHIFT_DEFAULT: float = 3000.0
GEAR_SPEED_CAPS: list[tuple[float, int]] = [(15.0, 1), (45.0, 2), (75.0, 3)]

# --------------- Safety / startup ---------------
STARTUP_STEPS:     int   = 120      # Reduced from 150 to exit startup sooner
FALLBACK_TRACKPOS: float = 1.2      # abs(trackPos) > this → recovery


class OptimalLineDriver(BaseDriver):
    """Track-position-indexed trajectory controller with late hard braking."""

    def __init__(self, map_path: Path | None = None) -> None:
        self._map_path    = map_path or _DEFAULT_MAP_PATH
        self._trajectory: Trajectory | None = None
        self._step_count: int   = 0
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

        # --- startup: full throttle, attenuated steer ---
        if self._step_count <= STARTUP_STEPS:
            steer = self._steer(state, 0.0) * 0.5
            self._prev_steer = steer
            return Action(steer=steer, accel=1.0, brake=0.0,
                          gear=self._startup_gear(state)).clamp()

        # --- off-track safety ---
        if abs(state.trackPos) > FALLBACK_TRACKPOS:
            return self._recovery(state)

        assert self._trajectory is not None
        traj = self._trajectory

        # current kinematic state
        v_cur = state.speed  # km/h

        # scan ahead for the nearest speed minimum
        v_min, d_to_min = traj.min_speed_ahead(state.distFromStart, SCAN_AHEAD_M)

        # --- decide: brake or accelerate? ---
        # d_onset = how far from the corner the car CAN WAIT before braking at BRAKE_MAX
        # v²_cur − v²_min = 2 × DECEL × d_onset  →  d_onset = (v²_cur − v²_min) / (2 × DECEL_MAX)
        accel: float = 0.0
        brake: float = 0.0

        decel_max_at_full_brake = BRAKE_MAX * BACKWARD_DECEL_FACTOR  # (km/h)²/m

        if v_min >= v_cur - 2.0:
            # minimum ahead is above (or close to) current speed → full throttle
            accel = 1.0
        else:
            d_onset = (v_cur ** 2 - v_min ** 2) / (2.0 * decel_max_at_full_brake)
            if d_to_min > d_onset + BRAKE_MARGIN_M:
                # Still enough room — go flat-out
                accel = 1.0
            else:
                # Time to brake: compute required pressure
                d_safe = max(1.0, d_to_min)
                required = (v_cur ** 2 - v_min ** 2) / (2.0 * d_safe)
                bp = min(BRAKE_MAX, required / BACKWARD_DECEL_FACTOR)
                brake = self._apply_abs(bp, state)
                accel = 0.0

        # --- if going slowly, ensure we're not coasting ---
        if brake == 0.0 and v_cur < 80.0:
            # In slow corners: boost throttle proportional to gap from target
            target_here = traj.lookup_speed(state.distFromStart)
            if v_cur < target_here:
                accel = max(accel, min(1.0, THROTTLE_BASE + (target_here - v_cur) * 0.02))

        # --- steering and gear ---
        target_tp = traj.lookup_line(state.distFromStart)
        steer = self._steer(state, target_tp)
        gear  = self._compute_gear(state, braking=(brake > 0))

        return Action(steer=steer, accel=accel, brake=brake, gear=gear).clamp()

    # ------------------------------------------------------------------
    def _steer(self, state: SensorState, target_trackPos: float) -> float:
        line_err = state.trackPos - target_trackPos
        raw = state.angle * STEER_ANGLE_GAIN - line_err * STEER_LINE_GAIN
        steer = raw / STEER_LOCK
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
        steer = -math.copysign(0.3, state.trackPos)
        return Action(steer=steer, accel=0.3, brake=0.0, gear=max(1, state.gear)).clamp()
