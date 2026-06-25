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
# Physics-based speed target
# ---------------------------------------------------------------------------
# Target is set to TARGET_PHYSICS_SCALE × physics equilibrium so braking always
# determines the corner speed rather than an arbitrary lookup table.
#
#   physics_safe(d) = sqrt(max(0, (d - BRAKE_MARGIN) * BRAKE_DECEL_FACTOR))
#
# BRAKE_DECEL_FACTOR calibrated for this simulator: gives ~1 g deceleration.
BRAKE_DECEL_FACTOR: float = 232.0
BRAKE_MARGIN: float = 5.0           # metres of safety headroom in stopping formula
TARGET_PHYSICS_SCALE: float = 1.20  # target = physics_safe × this → physics binds
MAX_SPEED: float = 200.0            # absolute cap (km/h)

# Edge speed limiters — only intervene when drifting near the absolute limit
EDGE_SPEED_SOFT: tuple[float, float] = (0.75, 140.0)
EDGE_SPEED_HARD: tuple[float, float] = (0.88, 100.0)

# ---------------------------------------------------------------------------
# Braking model
# ---------------------------------------------------------------------------
# Maximum brake pressure by speed regime
BRAKE_MAX_HIGH: float = 0.62   # > 140 km/h — partial; preserve steering authority
BRAKE_MAX_MED:  float = 0.78   # 90–140 km/h
BRAKE_MAX_LOW:  float = 0.90   # < 90 km/h

# Electronic Brake-force Distribution: back off while cornering
EBD_STEER_THRESH: float = 0.08
EBD_GAIN: float = 0.75
EBD_FLOOR: float = 0.40

# ---------------------------------------------------------------------------
# Full-throttle override
# ---------------------------------------------------------------------------
# Apply 100 % throttle when the forward sector sees at least this much clear road.
FULL_THROTTLE_LOOKAHEAD: float = 65.0

# ---------------------------------------------------------------------------
# Throttle PI (corner-exit and approach)
# ---------------------------------------------------------------------------
THROTTLE_KP: float = 0.40
THROTTLE_KI: float = 0.02
THROTTLE_MAX_INTEGRAL: float = 1.0

# ---------------------------------------------------------------------------
# Traction control (TCS)
# ---------------------------------------------------------------------------
TCS_STEER_THRESH: float = 0.14
TCS_GAIN_LOW_GEAR: float = 1.45  # gears 1–2
TCS_GAIN_MID_GEAR: float = 1.20  # gear 3
TCS_GAIN_HIGH_GEAR: float = 0.70 # gears 4+
TCS_MIN_ACCEL: float = 0.25

# Wheel-slip TCS
WHEEL_RADIUS: float = 0.33        # m — approximate TORCS tyre radius
TCS_SLIP_THRESHOLD: float = 1.25  # rear spin / expected > this triggers cut
TCS_SLIP_GAIN: float = 3.0

# ---------------------------------------------------------------------------
# Gear-shift RPM thresholds
# ---------------------------------------------------------------------------
RPM_UPSHIFT: float = 9000.0
RPM_DOWNSHIFT_BY_GEAR: dict[int, float] = {
    6: 6800.0,
    5: 6300.0,
    4: 5800.0,
    3: 4300.0,
    2: 3500.0,
}
RPM_DOWNSHIFT_DEFAULT: float = 3000.0

GEAR_SPEED_CAPS: list[tuple[float, int]] = [
    (15.0, 1),
    (45.0, 2),
    (75.0, 3),
]

# ---------------------------------------------------------------------------
# Startup / smoothing / recovery
# ---------------------------------------------------------------------------
STARTUP_STEPS: int = 80
STEER_SMOOTH_SPEED: float = 42.0
STEER_SMOOTH_ALPHA: float = 0.35

STUCK_TRACKPOS_THRESH: float = 0.9
STUCK_SPEED_THRESH: float = 5.0
STUCK_TIME_LIMIT: float = 3.0
REVERSE_DURATION: float = 2.0
STUCK_STARTUP_IMMUNITY: float = 6.0


class RuleBasedDriver(BaseDriver):
    """Physics-optimised steering + braking + RPM gearbox."""

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

    def on_restart(self) -> None:
        self.reset()

    def step(self, state: SensorState) -> Action:
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        dt = max(now - self._prev_time, 0.02) if self._prev_time else 0.02
        self._prev_time = now
        self._step_count += 1

        # Startup: attenuated steering, full throttle, speed-based gear
        if self._step_count <= STARTUP_STEPS:
            steer = self._compute_steering(state) * 0.5
            self._prev_steer = steer
            gear = self._startup_gear(state)
            return Action(steer=steer, accel=1.0, brake=0.0, gear=gear).clamp()

        # Recovery overrides normal driving
        if self._in_recovery(state, now):
            return self._recovery_action(state, now)

        steer = self._smooth_steer(self._compute_steering(state), state.speed)
        fwd_dist = self._fwd_dist(state)
        target_speed = self._target_speed(state, fwd_dist)
        physics_safe = _physics_safe_speed(fwd_dist)
        accel, brake = self._compute_throttle_brake(state, target_speed, physics_safe, fwd_dist, dt)
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
    # Forward distance — used for both speed target and throttle / brake
    # ------------------------------------------------------------------

    def _fwd_dist(self, state: SensorState) -> float:
        """Min of sensors 7–11 (≈ ±1–3°): the narrowest forward sector.

        Using the minimum means every central sensor must see clear road before
        we consider the path open; the maximum of the forward arc is deliberately
        NOT used so that a side sensor looking into the inside of a corner cannot
        inflate the estimated clear distance.
        """
        sensors = state.track
        if len(sensors) < 12:
            return 200.0
        return min(sensors[7:12])

    # ------------------------------------------------------------------
    # Speed target — physics-based, never lookup-table limited
    # ------------------------------------------------------------------

    def _target_speed(self, state: SensorState, fwd_dist: float) -> float:
        """Return TARGET_PHYSICS_SCALE × physics equilibrium speed.

        The physics equilibrium is the speed at which the stopping distance
        exactly equals fwd_dist.  Setting the target above equilibrium means
        braking (not the table) always limits corner speed — and there are no
        step discontinuities between breakpoints.
        """
        target = min(MAX_SPEED, _physics_safe_speed(fwd_dist) * TARGET_PHYSICS_SCALE)

        tp = abs(state.trackPos)
        if tp > EDGE_SPEED_HARD[0]:
            target = min(target, EDGE_SPEED_HARD[1])
        elif tp > EDGE_SPEED_SOFT[0]:
            target = min(target, EDGE_SPEED_SOFT[1])

        return target

    # ------------------------------------------------------------------
    # Throttle / brake
    # ------------------------------------------------------------------

    def _compute_throttle_brake(
        self,
        state: SensorState,
        target: float,
        physics_safe: float,
        fwd_dist: float,
        dt: float,
    ) -> tuple[float, float]:
        speed = state.speed
        stopping_dist = speed * speed / BRAKE_DECEL_FACTOR + BRAKE_MARGIN

        # PRIORITY 1 — Brake when wall is within stopping distance AND we are above
        # the physics-safe speed for that distance.
        if fwd_dist < stopping_dist and speed > physics_safe:
            diff = speed - physics_safe
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

        # PRIORITY 2 — Over target but no immediate braking needed: coast.
        if speed > target:
            diff = speed - target
            self._speed_integral = max(
                -THROTTLE_MAX_INTEGRAL,
                self._speed_integral - diff * dt,
            )
            return 0.0, 0.0

        # PRIORITY 3 — Below target and road is clear: flat out.
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

    # ------------------------------------------------------------------
    # Traction control
    # ------------------------------------------------------------------

    def _wheel_slip_factor(self, state: SensorState) -> float:
        if state.speed < 5.0 or len(state.wheelSpinVel) < 4:
            return 0.0
        speed_ms = state.speed * (1000.0 / 3600.0)
        expected = speed_ms / WHEEL_RADIUS
        if expected < 1.0:
            return 0.0
        rear_avg = (state.wheelSpinVel[2] + state.wheelSpinVel[3]) / 2.0
        return max(0.0, rear_avg / expected - TCS_SLIP_THRESHOLD)

    def _apply_tcs(self, steer: float, accel: float, state: SensorState) -> float:
        gear = state.gear

        excess = abs(steer) - TCS_STEER_THRESH
        if excess > 0.0:
            if gear <= 2:
                gain = TCS_GAIN_LOW_GEAR
            elif gear == 3:
                gain = TCS_GAIN_MID_GEAR
            else:
                gain = TCS_GAIN_HIGH_GEAR
            accel = min(accel, max(TCS_MIN_ACCEL, 1.0 - excess * gain))

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

        if state.rpm > RPM_UPSHIFT and gear < 6:
            gear += 1
        else:
            margin = 800.0 if braking else 0.0
            threshold = RPM_DOWNSHIFT_BY_GEAR.get(gear, RPM_DOWNSHIFT_DEFAULT)
            if state.rpm < (threshold - margin) and gear > 1:
                gear -= 1

        for speed_thresh, max_gear in GEAR_SPEED_CAPS:
            if state.speed < speed_thresh:
                gear = min(gear, max_gear)
                break

        return gear

    # ------------------------------------------------------------------
    # Stuck / recovery
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


# ---------------------------------------------------------------------------
# Module-level helper (no state, easy to unit-test)
# ---------------------------------------------------------------------------

def _physics_safe_speed(fwd_dist: float) -> float:
    """Speed (km/h) at which stopping distance equals fwd_dist.

    stopping_dist = v²/BRAKE_DECEL_FACTOR + BRAKE_MARGIN = fwd_dist
    → v = sqrt(max(0, (fwd_dist - BRAKE_MARGIN) * BRAKE_DECEL_FACTOR))
    """
    return math.sqrt(max(0.0, (fwd_dist - BRAKE_MARGIN) * BRAKE_DECEL_FACTOR))
