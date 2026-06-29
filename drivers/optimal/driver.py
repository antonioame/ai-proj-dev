"""OptimalLineDriver — safe steering + aggressive speed via map late-braking.

Design philosophy (rebuilt from scratch):
  * STEERING is conservative and proven — it mirrors the rule_based driver,
    which completes the lap with zero off-track excursions. Gentle apex-seeking
    only; no aggressive lock that darts the car off the road.
  * SPEED is where we PUSH. Because the track map knows exactly where every
    corner is (by distFromStart), we can:
      - run flat-out on straights up to a high cap,
      - brake as LATE as physics allows (not when a sensor trips),
      - carry slightly more apex speed than the recorded baseline.
  * Safety nets that never hurt the push: a live forward-sensor brake (in case
    the map is ever wrong), ABS, TCS, stuck-reverse, and anti-crawl.

The "push" knobs are grouped together below so they are easy to dial.

Requires: torcs_env/track_data/track_map.json
Build map: python scripts/build_track_map.py --telemetry data/<file>.csv
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from drivers.base_driver import BaseDriver
from drivers.optimal.trajectory import Trajectory
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from torcs_env.track_map import TrackMap

_DEFAULT_MAP_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "torcs_env" / "track_data" / "track_map.json"
)

# ===========================================================================
# PUSH knobs — raise these to go faster, lower them if it runs wide off a corner
# ===========================================================================
MAX_SPEED:          float = 250.0  # straight-line cap (rule_based uses 200)
CORNER_SPEED_SCALE: float = 1.08   # carry this × the recorded apex speed
BRAKE_DECEL:        float = 300.0  # (km/h)²/m — higher = brakes LATER (pushier)
BRAKE_MARGIN_M:     float = 6.0    # safety buffer before the latest brake point
SCAN_AHEAD_M:       float = 220.0  # how far ahead to look for the next corner

# ===========================================================================
# STEERING — conservative, copied from the proven rule_based driver
# ===========================================================================
STEER_ANGLE_GAIN:   float = 2.0
STEER_TRACK_GAIN:   float = 0.2
STEER_LOCK:         float = 0.785398   # 45° in radians
STEER_SMOOTH_SPEED: float = 42.0
STEER_SMOOTH_ALPHA: float = 0.35

# Gentle apex-seeking (same magnitude as rule_based — does NOT go off track)
APEX_CURV_GAIN: float = 0.30
APEX_TP_MAX:    float = 0.28

# ===========================================================================
# Braking model (per-speed max pressure, with ABS + cornering back-off)
# ===========================================================================
BRAKE_MAX_HIGH: float = 0.82   # > 140 km/h
BRAKE_MAX_MED:  float = 0.88   # 90–140 km/h
BRAKE_MAX_LOW:  float = 0.93   # < 90 km/h
EBD_STEER_THRESH: float = 0.08
EBD_GAIN: float = 0.75
EBD_FLOOR: float = 0.40

# Live forward sensor safety net: if the narrow forward sector sees less than
# this, brake regardless of the map (catches anything the map missed).
FWD_BRAKE_TRIGGER_M: float = 30.0

# ABS / TCS
WHEEL_RADIUS:        float = 0.33
ABS_SLIP_THRESHOLD:  float = 0.80
TCS_SLIP_THRESHOLD:  float = 1.25
TCS_SLIP_GAIN:       float = 3.0
TCS_MIN_ACCEL:       float = 0.20
TCS_STEER_THRESH:    float = 0.18

# Throttle
FULL_THROTTLE_LOOKAHEAD: float = 60.0
THROTTLE_KP: float = 0.40

# Gears
RPM_UPSHIFT:   float = 9000.0
RPM_DOWNSHIFT: dict[int, float] = {6: 6800, 5: 6300, 4: 5800, 3: 4300, 2: 3500}
RPM_DOWNSHIFT_DEFAULT: float = 3000.0
GEAR_SPEED_CAPS: list[tuple[float, int]] = [(15.0, 1), (45.0, 2), (75.0, 3)]

# Startup / safety
STARTUP_STEPS:      int   = 100
FALLBACK_TRACKPOS:  float = 1.05
CRAWL_SPEED:        float = 20.0

# Stuck detection
STUCK_SPEED_THRESH:     float = 5.0
STUCK_TIME_LIMIT:       float = 2.0
REVERSE_DURATION:       float = 1.5
STUCK_STARTUP_IMMUNITY: float = 6.0


class OptimalLineDriver(BaseDriver):
    """Safe steering, aggressive map-driven speed."""

    def __init__(self, map_path: Path | None = None) -> None:
        self._map_path = map_path or _DEFAULT_MAP_PATH
        self._trajectory: Trajectory | None = None
        self.reset()

    def _load_trajectory(self) -> None:
        track_map = TrackMap.load(self._map_path)
        self._trajectory = Trajectory.from_track_map(track_map)

    def reset(self) -> None:
        self._step_count = 0
        self._prev_steer = 0.0
        self._start_time: float | None = None
        self._stuck_since: float | None = None
        self._reversing_until: float | None = None

    def on_restart(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    def step(self, state: SensorState) -> Action:
        if self._trajectory is None:
            self._load_trajectory()

        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        self._step_count += 1

        # startup: gentle steer, full throttle, speed-capped gear
        if self._step_count <= STARTUP_STEPS:
            steer = self._steer(state, 0.0) * 0.5
            return Action(steer=steer, accel=1.0, brake=0.0,
                          gear=self._startup_gear(state)).clamp()

        # stuck against a wall → reverse out
        if self._handle_stuck(state, now):
            return self._reverse_action(state)

        # off-track but rolling → steer back, no braking
        if abs(state.trackPos) > FALLBACK_TRACKPOS:
            return self._recovery(state)

        # --- steering (safe) ---
        target_tp = self._apex_trackpos(state)
        steer = self._steer(state, target_tp)

        # --- speed (aggressive) ---
        accel, brake = self._speed_control(state)
        accel = self._apply_tcs(steer, accel, state)
        gear = self._compute_gear(state, braking=(brake > 0))

        return Action(steer=steer, accel=accel, brake=brake, gear=gear).clamp()

    # ------------------------------------------------------------------
    # Speed: flat-out until the latest safe braking point for the next corner
    # ------------------------------------------------------------------
    def _speed_control(self, state: SensorState) -> tuple[float, float]:
        assert self._trajectory is not None
        traj = self._trajectory
        v = state.speed

        # next corner from the map (do not look past the finish line)
        remaining = traj.track_length - (state.distFromStart % traj.track_length)
        scan = max(10.0, min(SCAN_AHEAD_M, remaining))
        v_min, d_to_min = traj.min_speed_ahead(state.distFromStart, scan)
        v_target_corner = max(CRAWL_SPEED, v_min * CORNER_SPEED_SCALE)

        # latest speed we may hold now and still brake down to the corner in time:
        #   v_allowed² = v_corner² + 2 · BRAKE_DECEL · (d − margin)
        d_brake = max(0.0, d_to_min - BRAKE_MARGIN_M)
        v_allowed = math.sqrt(v_target_corner ** 2 + 2.0 * BRAKE_DECEL * d_brake)
        target = min(MAX_SPEED, v_allowed)

        fwd = self._fwd_dist(state)

        # SAFETY NET: a genuinely close wall the map didn't account for
        if fwd < FWD_BRAKE_TRIGGER_M and v > v_target_corner:
            return 0.0, self._brake_pressure(state, v - v_target_corner)

        # Over the target → brake (this is the late-braking zone)
        if v > target + 1.0:
            return 0.0, self._brake_pressure(state, v - target)

        # At/under target with clear road → full throttle (PUSH)
        if fwd >= FULL_THROTTLE_LOOKAHEAD:
            return 1.0, 0.0

        # Below target, corner area → proportional throttle
        accel = min(1.0, max(0.0, THROTTLE_KP * (target - v) / 10.0 + 0.5))
        if v < CRAWL_SPEED:
            accel = max(accel, 0.6)  # anti-crawl
        return accel, 0.0

    def _brake_pressure(self, state: SensorState, overspeed: float) -> float:
        v = state.speed
        if v > 140.0:
            mx = BRAKE_MAX_HIGH
        elif v > 90.0:
            mx = BRAKE_MAX_MED
        else:
            mx = BRAKE_MAX_LOW
        # back off while cornering (electronic brake-force distribution)
        st = abs(self._prev_steer)
        if st > EBD_STEER_THRESH:
            mx = max(EBD_FLOOR, mx - (st - EBD_STEER_THRESH) * EBD_GAIN)
        brake = min(mx, overspeed / 10.0)
        return self._apply_abs(brake, state)

    # ------------------------------------------------------------------
    # Steering (conservative, proven)
    # ------------------------------------------------------------------
    def _apex_trackpos(self, state: SensorState) -> float:
        s = state.track
        if len(s) < 17:
            return 0.0
        left_avg = (s[2] + s[3] + s[4]) / 3.0
        right_avg = (s[14] + s[15] + s[16]) / 3.0
        total = left_avg + right_avg
        if total <= 1.0:
            return 0.0
        curvature = (left_avg - right_avg) / total
        return max(-APEX_TP_MAX, min(APEX_TP_MAX, -curvature * APEX_CURV_GAIN))

    def _steer(self, state: SensorState, target_tp: float) -> float:
        raw = (
            state.angle * STEER_ANGLE_GAIN
            - (state.trackPos - target_tp) * STEER_TRACK_GAIN
        )
        steer = raw / STEER_LOCK
        if state.speed < STEER_SMOOTH_SPEED:
            steer = (
                self._prev_steer * (1.0 - STEER_SMOOTH_ALPHA)
                + steer * STEER_SMOOTH_ALPHA
            )
        self._prev_steer = steer
        return steer

    def _fwd_dist(self, state: SensorState) -> float:
        if len(state.track) < 12:
            return 200.0
        return min(state.track[7:12])

    # ------------------------------------------------------------------
    # ABS / TCS
    # ------------------------------------------------------------------
    def _apply_abs(self, brake: float, state: SensorState) -> float:
        if brake < 0.05 or state.speed < 10.0 or len(state.wheelSpinVel) < 2:
            return brake
        expected = (state.speed / 3.6) / WHEEL_RADIUS
        if expected < 1.0:
            return brake
        front = (state.wheelSpinVel[0] + state.wheelSpinVel[1]) / 2.0
        ratio = front / expected
        if ratio < ABS_SLIP_THRESHOLD:
            lockup = ABS_SLIP_THRESHOLD - ratio
            brake = max(0.0, brake * (1.0 - lockup / ABS_SLIP_THRESHOLD))
        return brake

    def _apply_tcs(self, steer: float, accel: float, state: SensorState) -> float:
        # cut throttle when steering hard (avoid power understeer)
        excess = abs(steer) - TCS_STEER_THRESH
        if excess > 0.0:
            gain = 1.2 if state.gear <= 3 else 0.7
            accel = min(accel, max(TCS_MIN_ACCEL, 1.0 - excess * gain))
        # cut throttle on rear wheelspin
        if state.speed >= 5.0 and len(state.wheelSpinVel) >= 4:
            expected = (state.speed / 3.6) / WHEEL_RADIUS
            if expected >= 1.0:
                rear = (state.wheelSpinVel[2] + state.wheelSpinVel[3]) / 2.0
                slip = rear / expected - TCS_SLIP_THRESHOLD
                if slip > 0.0:
                    accel = min(accel, max(TCS_MIN_ACCEL, 1.0 - slip * TCS_SLIP_GAIN))
        return accel

    # ------------------------------------------------------------------
    # Gears
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Stuck / recovery
    # ------------------------------------------------------------------
    def _handle_stuck(self, state: SensorState, now: float) -> bool:
        elapsed = now - self._start_time if self._start_time is not None else 0.0
        if elapsed < STUCK_STARTUP_IMMUNITY:
            return False
        if self._reversing_until is not None:
            if now < self._reversing_until:
                return True
            self._reversing_until = None
            self._stuck_since = None
            return False
        pinned = abs(state.trackPos) > 0.9 and state.speed < STUCK_SPEED_THRESH
        if pinned:
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since > STUCK_TIME_LIMIT:
                self._reversing_until = now + REVERSE_DURATION
                return True
        else:
            self._stuck_since = None
        return False

    def _reverse_action(self, state: SensorState) -> Action:
        steer = -math.copysign(0.5, state.trackPos)
        self._prev_steer = 0.0
        return Action(steer=steer, accel=0.3, brake=0.0, gear=-1).clamp()

    def _recovery(self, state: SensorState) -> Action:
        steer = -math.copysign(0.6, state.trackPos)
        return Action(steer=steer, accel=0.45, brake=0.0,
                      gear=max(1, state.gear)).clamp()
