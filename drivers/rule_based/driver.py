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
# Effective gain = STEER_ANGLE_GAIN / STEER_LOCK.
# Standard SCR bots use angle*10/PI ≈ angle*3.18; we target ~2.5 effective gain.
STEER_ANGLE_GAIN: float = 2.0      # raised from 1.0 — faster heading correction
STEER_TRACK_GAIN: float = 0.2      # lowered from 0.3 — softer centre-return
STEER_LOCK: float = 0.785398       # 45° in radians

# ---------------------------------------------------------------------------
# Speed targets via lookahead (max of forward-arc sensors 5–13, ≈ -10° to +10°)
# Each tuple is (min_lookahead_metres, target_speed_km/h)
# ---------------------------------------------------------------------------
LOOKAHEAD_SPEEDS: list[tuple[float, float]] = [
    (80.0, 120.0),   # open road — up to 120 km/h
    (50.0,  80.0),   # medium corner approach — 80 km/h
    (30.0,  55.0),   # tight section — 55 km/h
    ( 0.0,  38.0),   # very short sightline — 38 km/h
]

# Reduce target when car drifts toward the edge
EDGE_SPEED_SOFT: tuple[float, float] = (0.65, 80.0)   # (|trackPos| threshold, max km/h)
EDGE_SPEED_HARD: tuple[float, float] = (0.78, 55.0)

# ---------------------------------------------------------------------------
# Throttle / brake PI gains
# ---------------------------------------------------------------------------
THROTTLE_KP: float = 0.30
THROTTLE_KI: float = 0.02
THROTTLE_MAX_INTEGRAL: float = 1.0

# ---------------------------------------------------------------------------
# Traction control (TCS)
# ---------------------------------------------------------------------------
TCS_STEER_THRESH: float = 0.10    # |normalised steer| above this triggers TCS
TCS_GAIN: float = 1.2             # how quickly TCS cuts throttle beyond threshold
TCS_MIN_ACCEL: float = 0.18       # floor so car keeps moving

# ---------------------------------------------------------------------------
# Gear-shift RPM thresholds
# ---------------------------------------------------------------------------
RPM_UPSHIFT: float = 8000.0       # raised from 6500 — use more of the power band
RPM_DOWNSHIFT: float = 3000.0

# Speed-based gear cap: prevents being in too high a gear at low speed
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
# Steering low-pass filter
# ---------------------------------------------------------------------------
STEER_SMOOTH_SPEED: float = 42.0  # apply smoothing below this speed (km/h)
STEER_SMOOTH_ALPHA: float = 0.35  # weight of new value (1-alpha = weight of previous)

# ---------------------------------------------------------------------------
# Recovery / stuck detection
# ---------------------------------------------------------------------------
STUCK_TRACKPOS_THRESH: float = 0.9
STUCK_SPEED_THRESH: float = 5.0
STUCK_TIME_LIMIT: float = 3.0
REVERSE_DURATION: float = 2.0
# Seconds from race start before stuck detection is armed (avoids false trigger at launch)
STUCK_STARTUP_IMMUNITY: float = 6.0


class RuleBasedDriver(BaseDriver):
    """Proportional steering + lookahead speed control + PI throttle + RPM gearbox."""

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

        # -- Startup phase: gentle steering, full throttle, simple gear logic --
        if self._step_count <= STARTUP_STEPS:
            steer = self._compute_steering(state) * 0.5  # attenuate only; no extra smoothing
            self._prev_steer = steer
            gear = self._startup_gear(state)
            return Action(steer=steer, accel=1.0, brake=0.0, gear=gear).clamp()

        # -- Recovery mode overrides normal driving --
        if self._in_recovery(state, now):
            return self._recovery_action(state, now)

        steer = self._smooth_steer(self._compute_steering(state), state.speed)
        target_speed = self._target_speed(state)
        accel, brake = self._compute_throttle_brake(state, target_speed, dt)
        accel = self._apply_tcs(steer, accel)
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
        return steer / STEER_LOCK

    def _smooth_steer(self, steer: float, speed: float) -> float:
        """Low-pass filter at low speed to damp oscillations."""
        if speed < STEER_SMOOTH_SPEED:
            steer = self._prev_steer * (1.0 - STEER_SMOOTH_ALPHA) + steer * STEER_SMOOTH_ALPHA
        self._prev_steer = steer
        return steer

    # ------------------------------------------------------------------
    # Speed target via lookahead
    # ------------------------------------------------------------------

    def _lookahead(self, state: SensorState) -> float:
        """Max track sensor reading in the forward arc (sensors 5–13, ≈ ±10°)."""
        sensors = state.track
        if len(sensors) < 14:
            return 200.0
        return max(sensors[5:14])

    def _target_speed(self, state: SensorState) -> float:
        ahead = self._lookahead(state)
        for thresh, speed in LOOKAHEAD_SPEEDS:
            if ahead > thresh:
                target = speed
                break
        else:
            target = 38.0

        # Reduce target near track edges
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

    def _apply_tcs(self, steer: float, accel: float) -> float:
        """Reduce throttle when turning to prevent wheelspin."""
        excess = abs(steer) - TCS_STEER_THRESH
        if excess > 0.0:
            max_accel = max(TCS_MIN_ACCEL, 1.0 - excess * TCS_GAIN)
            return min(accel, max_accel)
        return accel

    # ------------------------------------------------------------------
    # Gear shifting
    # ------------------------------------------------------------------

    def _startup_gear(self, state: SensorState) -> int:
        """Simple speed-based gear for the launch phase."""
        v = state.speed
        if v < 15.0:
            return 1
        if v < 45.0:
            return 2
        return 3

    def _compute_gear(self, state: SensorState) -> int:
        gear = state.gear if state.gear >= 1 else 1
        if state.rpm > RPM_UPSHIFT and gear < 6:
            gear += 1
        elif state.rpm < RPM_DOWNSHIFT and gear > 1:
            gear -= 1
        # Prevent being in too high a gear at low speed
        for speed_thresh, max_gear in GEAR_SPEED_CAPS:
            if state.speed < speed_thresh:
                gear = min(gear, max_gear)
                break
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
        # Suppress during the launch window to avoid false positives at speed=0
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
        """Drive in reverse toward track centre."""
        steer = -math.copysign(0.5, state.trackPos)
        return Action(
            steer=steer,
            accel=0.3,
            brake=0.0,
            gear=-1,
            clutch=0.0,
        ).clamp()
