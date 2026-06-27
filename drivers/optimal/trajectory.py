"""Speed profile and racing line trajectory derived from a TrackMap.

Design:
  * Straight buckets   → MAX_SPEED_KMH (flat-out)
  * Corner buckets     → b.speed_kmh (= the telemetry MINIMUM per bucket)
                         optionally scaled by CORNER_SPEED_SCALE
  * Backward pass      → propagates each corner speed backward so the car
                         can stop in time; determines the exact braking point.

With the telemetry minimum per bucket as the corner floor, the trajectory
preserves the correct braking profile through multi-apex complexes while
allowing the car to go flat-out on genuine straights.  The gain vs the
reactive rule_based driver is late braking: the trajectory knows EXACTLY
where each corner is via distFromStart, so it brakes far later.
"""
from __future__ import annotations

import math
from torcs_env.track_map import TrackMap, BUCKET_M, TRACK_LENGTH_M

MAX_SPEED_KMH: float   = 200.0
MIN_SPEED_KMH: float   = 10.0
CORNER_SPEED_SCALE: float = 1.0    # 1.0 = same as telemetry floor; >1 pushes corners harder
BACKWARD_DECEL_FACTOR: float = 270.0  # (km/h)²/m
MAX_BACKWARD_PASSES: int = 100


class Trajectory:
    """Speed profile + racing line, indexed by distFromStart."""

    def __init__(
        self,
        speed_profile: list[float],
        line_profile: list[float],
        track_length: float = TRACK_LENGTH_M,
    ) -> None:
        self._speed = speed_profile
        self._line  = line_profile
        self._n     = len(speed_profile)
        self.track_length = track_length

    # ------------------------------------------------------------------
    def _idx(self, s: float) -> int:
        return min(int(s % self.track_length / BUCKET_M), self._n - 1)

    def lookup_speed(self, s: float) -> float:
        return self._speed[self._idx(s)]

    def lookup_line(self, s: float) -> float:
        return self._line[self._idx(s)]

    def min_speed_ahead(self, s: float, scan_m: float) -> tuple[float, float]:
        """Return (min_speed_kmh, dist_to_min_m) in the window [s, s+scan_m]."""
        steps = max(1, int(scan_m / BUCKET_M))
        min_v = float("inf")
        min_d = scan_m
        for k in range(steps + 1):
            d = k * BUCKET_M
            v = self._speed[self._idx(s + d)]
            if v < min_v:
                min_v = v
                min_d = d
        return min_v, min_d

    # ------------------------------------------------------------------
    @classmethod
    def from_track_map(cls, track_map: TrackMap) -> "Trajectory":
        n = len(track_map.buckets)

        # --- step 1: set floors ---
        # Straights: MAX_SPEED (controller will be flat-out)
        # Corners:   telemetry minimum per bucket × scale
        speed_profile: list[float] = []
        for b in track_map.buckets:
            if b.is_corner:
                speed_profile.append(
                    max(MIN_SPEED_KMH, b.speed_kmh * CORNER_SPEED_SCALE)
                )
            else:
                speed_profile.append(MAX_SPEED_KMH)

        # --- step 2: backward pass (run to convergence) ---
        # Iterates from i=n-1 down to 0; using the already-updated [i+1] value
        # means a single pass propagates constraints all the way back.
        # Two passes handle the finish-line wrap.
        ds = BUCKET_M
        for _ in range(MAX_BACKWARD_PASSES):
            changed = False
            for i in reversed(range(n)):
                next_i = (i + 1) % n
                v_max = math.sqrt(
                    speed_profile[next_i] ** 2 + BACKWARD_DECEL_FACTOR * 2.0 * ds
                )
                if speed_profile[i] > v_max + 0.01:
                    speed_profile[i] = v_max
                    changed = True
            if not changed:
                break

        # --- step 3: clamp ---
        speed_profile = [
            max(MIN_SPEED_KMH, min(MAX_SPEED_KMH, v)) for v in speed_profile
        ]

        # --- racing line from telemetry ---
        line_profile: list[float] = [b.trackPos for b in track_map.buckets]

        return cls(speed_profile, line_profile, track_map.track_length)
