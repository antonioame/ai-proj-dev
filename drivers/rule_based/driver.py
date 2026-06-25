"""Rule-based baseline driver for TORCS / Corkscrew.

Tuning constants are defined at module level so they are easy to adjust.
"""

from __future__ import annotations

import math
import time

from drivers.base_driver import BaseDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState

# ---------------------------------------------------------------------------
# Steering constants
# ---------------------------------------------------------------------------
STEER_ANGLE_GAIN: float = 2.0
STEER_TRACK_GAIN: float = 0.2
STEER_LOCK: float = 0.785398  # 45° in radians

# ---------------------------------------------------------------------------
# Speed targets (km/h) via lookahead — aggressive racing targets
# Each tuple: (min_lookahead_metres, target_speed_km/h)
# ---------------------------------------------------------------------------
LOOKAHEAD_SPEEDS: list[tuple[float, float]] = [
    (170.0, 200.0),   # long straight — flat out
    (120.0, 185.0),   # gentle open curve
    ( 80.0, 155.0),   # moderate curve
    ( 50.0, 115.0),   # medium corner
    ( 30.0,  85.0),   # tight corner
    (  0.0,  62.0),   # hairpin / very tight
]

# Edge speed limiters — only intervene near the absolute limit
EDGE_SPEED_SOFT: tuple[float, float] = (0.75, 140.0)
EDGE_SPEED_HARD: tuple[float, float] = (0.88, 100.0)

# ---------------------------------------------------------------------------
# Full-throttle override
# ---------------------------------------------------------------------------
# Apply 100% throttle when the forward road is this clear and we are below target.
FULL_THROTTLE_LOOKAHEAD: float = 80.0

# ---------------------------------------------------------------------------
# Braking model
# ---------------------------------------------------------------------------
# Stopping distance estimate (speed in km/h, distance in metres):
#   dist = speed² / BRAKE_DECEL_FACTOR + BRAKE_MARGIN
# Calibrated from the reference implementation used in the same simulator.
BRAKE_DECEL_FACTOR: float = 232.0
BRAKE_MARGIN: float = 5.0

# Maximum brake pressure by speed regime
BRAKE_MAX_HIGH: float = 0.62   # > 140 km/h — partial braking to preserve steering
BRAKE_MAX_MED:  float = 0.78   # 90–140 km/h
BRAKE_MAX_LOW:  float = 0.90   # < 90 km/h — firm braking

# Electronic Brake-force Distribution: reduce braking while steering
EBD_STEER_THRESH: float = 0.08
EBD_GAIN: float = 0.75
EBD_FLOOR: float = 0.40

# ---------------------------------------------------------------------------
# Throttle PI (used below FULL_THROTTLE_LOOKAHEAD and on corner exit)
# ---------------------------------------------------------------------------
THROTTLE_KP: float = 0.40
THROTTLE_KI: float = 0.02
THROTTLE_MAX_INTEGRAL: float = 1.0

# ---------------------------------------------------------------------------
# Traction control (TCS)
# ---------------------------------------------------------------------------
TCS_STEER_THRESH: float = 0.14   # normalised steer above which steer-TCS intervenes
TCS_GAIN_LOW_GEAR: float = 1.45  # gears 1–2: high torque, needs more cut
TCS_GAIN_MID_GEAR: float = 1.20  # gear 3
TCS_GAIN_HIGH_GEAR: float = 0.70 # gears 4+: minimal intervention
TCS_MIN_ACCEL: float = 0.25      # always allow some throttle

# Wheel-slip TCS: ratio of rear wheel spin to expected spin before cutting throttle
WHEEL_RADIUS: float = 0.33       # approximate TORCS wheel radius (m)
TCS_SLIP_THRESHOLD: float = 1.25 # rear spinning 25% faster than expected → intervene
TCS_SLIP_GAIN: float = 3.0       # throttle cut per unit of excess slip ratio

# ---------------------------------------------------------------------------
# Gear-shift RPM thresholds
# ---------------------------------------------------------------------------
RPM_UPSHIFT: float = 9000.0          # higher in the power band
RPM_DOWNSHIFT_BY_GEAR: dict[int, float] = {
    6: 6800.0,
    5: 6300.0,
    4: 5800.0,
    3: 4300.0,
    2: 3500.0,
}
RPM_DOWNSHIFT_DEFAULT: float = 3000.0

# Speed-based gear floor (prevents stalling at very low speed)
GEAR_SPEED_CAPS: list[tuple[float, int]] = [
    (15.0, 1),
    (45.0, 2),
    (75.0, 3),
]

# ---------------------------------------------------------------------------
# Startup phase
# ---------------------------------------------------------------------------
STARTUP_STEPS: int = 80           # first N steps: attenuated steering, full throttle

# ---------------------------------------------------------------------------
# Steering low-pass filter (low-speed only)
# ---------------------------------------------------------------------------
STEER_SMOOTH_SPEED: float = 42.0
STEER_SMOOTH_ALPHA: float = 0.35  # weight of new value; 1-alpha = weight of previous

# ---------------------------------------------------------------------------
# Recovery / stuck detection
# ---------------------------------------------------------------------------
STUCK_TRACKPOS_THRESH: float = 0.9
STUCK_SPEED_THRESH: float = 5.0
STUCK_TIME_LIMIT: float = 3.0
REVERSE_DURATION: float = 2.0
STUCK_STARTUP_IMMUNITY: float = 6.0


class RuleBasedDriver(BaseDriver):
    """Proportional steering + lookahead speed control + physics braking + RPM gearbox."""

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # BaseDriver interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._speed_integral: float = 0.0
        self._prev_time: float = 0.0
        self._prev_steer: float = 0.0
        self._step_count: int = 0
        self._start_time: float | None = None
        self._stuck_since: float | None = None
        self._reversing_until: float | None = None
        self._gear: int = 1  # remembered across steps for smooth shifting

    def on_restart(self) -> None:
        self.reset()

    def step(self, state: SensorState) -> Action:
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        dt = max(now - self._prev_time, 0.02) if self._prev_time else 0.02
        self._prev_time = now
        self._step_count += 1

        # Startup: attenuated steering, full throttle, simple gear logic
        if self._step_count <= STARTUP_STEPS:
            steer = self._compute_steering(state) * 0.5
            self._prev_steer = steer
            gear = self._startup_gear(state)
            self._gear = gear
            return Action(steer=steer, accel=1.0, brake=0.0, gear=gear).clamp()

        # Recovery overrides normal driving
        if self._in_recovery(state, now):
            return self._recovery_action(state, now)

        steer = self._smooth_steer(self._compute_steering(state), state.speed)
        lookahead = self._lookahead(state)
        fwd_dist = self._fwd_dist(state)
        # Conservative effective lookahead: use the shorter of the wide arc and
        # the central forward sector so that the target speed is never set by a
        # side sensor pointing into the inside of a corner.
        effective_lookahead = min(lookahead, fwd_dist)
        target_speed = self._target_speed(state, effective_lookahead)
        accel, brake = self._compute_throttle_brake(state, target_speed, fwd_dist, dt)
        accel = self._apply_tcs(steer, accel, state)
        gear = self._compute_gear(state, brake > 0.0)

        return Action(steer=steer, accel=accel, brake=brake, gear=gear).clamp()

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    def _compute_steering(self, state: SensorState) -> float:
        steer = (
            state.angle * STEER_ANGLE_GAIN
            - state.trackPos * STEER_TRACK_GAIN
        )
        return steer / STEER_LOCK

    def _smooth_steer(self, steer: float, speed: float) -> float:
        if speed < STEER_SMOOTH_SPEED:
            steer = self._prev_steer * (1.0 - STEER_SMOOTH_ALPHA) + steer * STEER_SMOOTH_ALPHA
        self._prev_steer = steer
        return steer

    # ------------------------------------------------------------------
    # Speed target via lookahead
    # ------------------------------------------------------------------

    def _lookahead(self, state: SensorState) -> float:
        """Max track-sensor reading across the forward arc (sensors 5–13, ≈ ±10°).
        Used for speed target selection."""
        sensors = state.track
        if len(sensors) < 14:
            return 200.0
        return max(sensors[5:14])

    def _fwd_dist(self, state: SensorState) -> float:
        """Minimum forward-sector reading (sensors 7–11, ≈ ±3°–6°).
        Used for braking and full-throttle decisions: must see clear road in all central
        directions before going flat out or deferring brakes."""
        sensors = state.track
        if len(sensors) < 12:
            return 200.0
        return min(sensors[7:12])

    def _target_speed(self, state: SensorState, lookahead: float) -> float:
        for thresh, speed in LOOKAHEAD_SPEEDS:
            if lookahead > thresh:
                target = speed
                break
        else:
            target = 62.0

        tp = abs(state.trackPos)
        soft_thresh, soft_speed = EDGE_SPEED_SOFT
        hard_thresh, hard_speed = EDGE_SPEED_HARD
        if tp > hard_thresh:
            target = min(target, hard_speed)
        elif tp > soft_thresh:
            target = min(target, soft_speed)

        return target

    # ------------------------------------------------------------------
    # Throttle / brake
    # ------------------------------------------------------------------

    def _compute_throttle_brake(
        self, state: SensorState, target: float, fwd_dist: float, dt: float
    ) -> tuple[float, float]:
        speed = state.speed
        stopping_dist = speed * speed / BRAKE_DECEL_FACTOR + BRAKE_MARGIN

        # PRIORITY 1 — Brake when wall is within stopping distance AND we are over target.
        # Braking while at or below target would slow the car unnecessarily.
        if fwd_dist < stopping_dist and speed > target:
            diff = speed - target
            if speed > 140.0:
                max_brake = BRAKE_MAX_HIGH
            elif speed > 90.0:
                max_brake = BRAKE_MAX_MED
            else:
                max_brake = BRAKE_MAX_LOW
            steer_abs = abs(self._prev_steer)
            if steer_abs > EBD_STEER_THRESH:
                max_brake = max(EBD_FLOOR, max_brake - (steer_abs - EBD_STEER_THRESH) * EBD_GAIN)
            brake = min(max_brake, diff / 10.0)
            self._speed_integral = 0.0
            return 0.0, brake

        # PRIORITY 2 — Over target but road is clear enough to coast.
        if speed > target:
            diff = speed - target
            self._speed_integral = max(
                -THROTTLE_MAX_INTEGRAL,
                self._speed_integral - diff * dt,
            )
            return 0.0, 0.0

        # PRIORITY 3 — Below target and all central sensors see open road: flat out.
        if fwd_dist >= FULL_THROTTLE_LOOKAHEAD:
            self._speed_integral = min(
                THROTTLE_MAX_INTEGRAL,
                self._speed_integral + (target - speed) * dt,
            )
            return 1.0, 0.0

        # PRIORITY 4 — Below target but corner ahead: proportional + integral throttle.
        error = target - speed
        self._speed_integral = max(
            -THROTTLE_MAX_INTEGRAL,
            min(THROTTLE_MAX_INTEGRAL, self._speed_integral + error * dt),
        )
        control = THROTTLE_KP * error + THROTTLE_KI * self._speed_integral
        return min(control, 1.0), 0.0

    def _wheel_slip_factor(self, state: SensorState) -> float:
        """Excess rear-wheel spin ratio (0 = no slip, positive = spinning)."""
        if state.speed < 5.0 or len(state.wheelSpinVel) < 4:
            return 0.0
        speed_ms = state.speed * (1000.0 / 3600.0)
        expected = speed_ms / WHEEL_RADIUS
        if expected < 1.0:
            return 0.0
        rear_avg = (state.wheelSpinVel[2] + state.wheelSpinVel[3]) / 2.0
        return max(0.0, rear_avg / expected - TCS_SLIP_THRESHOLD)

    def _apply_tcs(self, steer: float, accel: float, state: SensorState) -> float:
        """Combined TCS: steer-based (cornering) + wheel-slip (wheelspin)."""
        gear = state.gear

        # Steer-based cut: reduce throttle when cornering hard
        excess = abs(steer) - TCS_STEER_THRESH
        if excess > 0.0:
            if gear <= 2:
                gain = TCS_GAIN_LOW_GEAR
            elif gear == 3:
                gain = TCS_GAIN_MID_GEAR
            else:
                gain = TCS_GAIN_HIGH_GEAR
            accel = min(accel, max(TCS_MIN_ACCEL, 1.0 - excess * gain))

        # Wheel-slip cut: reduce throttle when rear wheels spin faster than expected
        slip = self._wheel_slip_factor(state)
        if slip > 0.0:
            accel = min(accel, max(TCS_MIN_ACCEL, 1.0 - slip * TCS_SLIP_GAIN))

        return accel

    # ------------------------------------------------------------------
    # Gear shifting
    # ------------------------------------------------------------------

    def _startup_gear(self, state: SensorState) -> int:
        v = state.speed
        if v < 15.0:
            return 1
        if v < 45.0:
            return 2
        return 3

    def _compute_gear(self, state: SensorState, braking: bool) -> int:
        gear = state.gear if state.gear >= 1 else 1

        # Upshift
        if state.rpm > RPM_UPSHIFT and gear < 6:
            gear += 1
        # Downshift — per-gear thresholds with extra margin while braking
        else:
            margin = 800.0 if braking else 0.0
            threshold = RPM_DOWNSHIFT_BY_GEAR.get(gear, RPM_DOWNSHIFT_DEFAULT)
            if state.rpm < (threshold - margin) and gear > 1:
                gear -= 1

        # Speed-based floor
        for speed_thresh, max_gear in GEAR_SPEED_CAPS:
            if state.speed < speed_thresh:
                gear = min(gear, max_gear)
                break

        self._gear = gear
        return gear

    # ------------------------------------------------------------------
    # Stuck / recovery detection
    # ------------------------------------------------------------------

    def _is_stuck(self, state: SensorState) -> bool:
        return (
            abs(state.trackPos) > STUCK_TRACKPOS_THRESH
            or state.speed < STUCK_SPEED_THRESH
        )

    def _in_recovery(self, state: SensorState, now: float) -> bool:
        elapsed = now - self._start_time if self._start_time is not None else 0.0
        if elapsed < STUCK_STARTUP_IMMUNITY:
            return False

        if self._reversing_until is not None and now < self._reversing_until:
            return True

        if self._is_stuck(state):
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since > STUCK_TIME_LIMIT:
                self._reversing_until = now + REVERSE_DURATION
                self._stuck_since = None
                self._speed_integral = 0.0
                return True
        else:
            self._stuck_since = None
            if self._reversing_until is not None and now >= self._reversing_until:
                self._reversing_until = None

        return False

    def _recovery_action(self, state: SensorState, now: float) -> Action:
        steer = -math.copysign(0.5, state.trackPos)
        return Action(
            steer=steer,
            accel=0.3,
            brake=0.0,
            gear=-1,
            clutch=0.0,
        ).clamp()
