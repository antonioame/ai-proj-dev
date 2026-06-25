"""Rule-based baseline driver for TORCS / Corkscrew.

Tuning constants are defined at module level so they are easy to adjust.
All control logic is documented inline so the design intent is clear.
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
STEER_ANGLE_GAIN: float = 1.0      # k1: how strongly to correct car angle
STEER_TRACK_GAIN: float = 0.3      # k2: how strongly to correct track-centre offset
STEER_LOCK: float = 0.785398       # maximum steer magnitude (45° in radians)

# ---------------------------------------------------------------------------
# Speed targets (km/h)
# ---------------------------------------------------------------------------
SPEED_STRAIGHT: float = 120.0      # target speed on a straight
SPEED_MEDIUM_CURVE: float = 80.0   # medium-curvature corner
SPEED_SHARP_CURVE: float = 50.0    # tight corner

# Curvature thresholds (difference between symmetric track sensors, metres)
CURVE_MEDIUM_THRESH: float = 30.0
CURVE_SHARP_THRESH: float = 60.0

# ---------------------------------------------------------------------------
# Throttle / brake PI gains
# ---------------------------------------------------------------------------
THROTTLE_KP: float = 0.25          # proportional gain for speed error
THROTTLE_KI: float = 0.02          # integral gain (prevents steady-state error)
THROTTLE_MAX_INTEGRAL: float = 1.0

# ---------------------------------------------------------------------------
# Gear-shift RPM thresholds
# ---------------------------------------------------------------------------
RPM_UPSHIFT: float = 6500.0
RPM_DOWNSHIFT: float = 3000.0

# ---------------------------------------------------------------------------
# Recovery / stuck detection
# ---------------------------------------------------------------------------
STUCK_TRACKPOS_THRESH: float = 0.9   # |trackPos| above this → consider stuck
STUCK_SPEED_THRESH: float = 5.0      # km/h — below this counts as low-speed
STUCK_TIME_LIMIT: float = 3.0        # seconds before triggering reverse
REVERSE_DURATION: float = 2.0        # seconds to reverse before re-attempting


class RuleBasedDriver(BaseDriver):
    """Simple proportional steering + PI speed control + RPM-based gearbox."""

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # BaseDriver interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._speed_integral: float = 0.0
        self._prev_time: float = 0.0
        self._stuck_since: float | None = None
        self._reversing_until: float | None = None

    def on_restart(self) -> None:
        self.reset()

    def step(self, state: SensorState) -> Action:
        now = time.monotonic()
        dt = max(now - self._prev_time, 0.02) if self._prev_time else 0.02
        self._prev_time = now

        # Recovery mode overrides normal driving
        if self._in_recovery(state, now):
            return self._recovery_action(state, now)

        steer = self._compute_steering(state)
        target_speed = self._target_speed(state)
        accel, brake = self._compute_throttle_brake(state, target_speed, dt)
        gear = self._compute_gear(state)

        return Action(steer=steer, accel=accel, brake=brake, gear=gear).clamp()

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    def _compute_steering(self, state: SensorState) -> float:
        """P controller: correct heading angle + track-centre offset."""
        steer = (
            state.angle * STEER_ANGLE_GAIN
            - state.trackPos * STEER_TRACK_GAIN
        )
        # Normalise by the steering lock so the output is in [-1, 1]
        return steer / STEER_LOCK

    # ------------------------------------------------------------------
    # Speed target
    # ------------------------------------------------------------------

    def _curvature(self, state: SensorState) -> float:
        """Estimate local curvature from the asymmetry of the track sensors.

        The 19 sensors are symmetric around index 9 (pointing straight ahead).
        If the left-side sensors are much shorter than the right-side sensors
        the car is in a right-hand curve, and vice versa.  We use the max
        of the symmetric differences as a curvature proxy.
        """
        sensors = state.track
        if len(sensors) < 19:
            return 0.0

        diffs = [abs(sensors[9 - i] - sensors[9 + i]) for i in range(1, 10)]
        return max(diffs)

    def _target_speed(self, state: SensorState) -> float:
        curvature = self._curvature(state)
        if curvature > CURVE_SHARP_THRESH:
            return SPEED_SHARP_CURVE
        if curvature > CURVE_MEDIUM_THRESH:
            return SPEED_MEDIUM_CURVE
        return SPEED_STRAIGHT

    # ------------------------------------------------------------------
    # Throttle / brake
    # ------------------------------------------------------------------

    def _compute_throttle_brake(
        self, state: SensorState, target: float, dt: float
    ) -> tuple[float, float]:
        error = target - state.speed
        self._speed_integral = max(
            -THROTTLE_MAX_INTEGRAL,
            min(THROTTLE_MAX_INTEGRAL, self._speed_integral + error * dt),
        )
        control = THROTTLE_KP * error + THROTTLE_KI * self._speed_integral

        if control >= 0.0:
            return min(control, 1.0), 0.0
        else:
            return 0.0, min(-control, 1.0)

    # ------------------------------------------------------------------
    # Gear shifting
    # ------------------------------------------------------------------

    def _compute_gear(self, state: SensorState) -> int:
        gear = state.gear if state.gear >= 1 else 1
        if state.rpm > RPM_UPSHIFT and gear < 6:
            return gear + 1
        if state.rpm < RPM_DOWNSHIFT and gear > 1:
            return gear - 1
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
        if self._reversing_until is not None and now < self._reversing_until:
            return True

        if self._is_stuck(state):
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since > STUCK_TIME_LIMIT:
                # Trigger reverse
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
        """Drive in reverse toward track centre."""
        # Steer toward centre: if trackPos > 0 (left of centre) steer right
        steer = -math.copysign(0.5, state.trackPos)
        return Action(
            steer=steer,
            accel=0.3,
            brake=0.0,
            gear=-1,
            clutch=0.0,
        ).clamp()
