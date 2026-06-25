"""Parse SCR sensor strings into a typed dataclass.

SCR sensor strings look like:
  (angle 0.1)(speedX 50.2)(trackPos 0.0)(track 200 180 ...)(rpm 4500)...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Regex: matches (key val1 val2 ...) tokens
_TOKEN_RE = re.compile(r'\((\w+)\s+([^)]+)\)')


def _floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split()]


def _int(raw: str) -> int:
    return int(float(raw))


@dataclass
class SensorState:
    # Car orientation vs track axis (radians, positive = pointing left)
    angle: float = 0.0

    # Longitudinal / lateral / vertical speed (km/h)
    speed: float = 0.0
    speedY: float = 0.0
    speedZ: float = 0.0

    # Track position: 0 = centre, ±1 = edge, > ±1 = off-track
    trackPos: float = 0.0

    # 19 range-finder readings (metres, 200 m max), evenly spaced -90° to +90°
    track: list[float] = field(default_factory=lambda: [200.0] * 19)

    # 36 opponent distance sensors (metres, 200 m max)
    opponents: list[float] = field(default_factory=lambda: [200.0] * 36)

    rpm: float = 0.0
    gear: int = 0
    damage: float = 0.0

    # Distance covered since race start (metres)
    distRaced: float = 0.0
    distFromStart: float = 0.0

    # Lap counter derived from distRaced resets (set externally by client)
    lap: int = 1

    lastLapTime: float = 0.0
    curLapTime: float = 0.0
    racePos: int = 1
    fuel: float = 94.0

    # Four wheel spin velocities (rad/s)
    wheelSpinVel: list[float] = field(default_factory=lambda: [0.0] * 4)

    # Car height above track surface (metres)
    z: float = 0.0

    # Raw string (useful for debugging)
    raw: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_string(cls, sensor_str: str) -> "SensorState":
        """Parse a raw SCR sensor string into a SensorState."""
        state = cls(raw=sensor_str)
        tokens = _TOKEN_RE.findall(sensor_str)

        for key, val in tokens:
            val = val.strip()
            if key == "angle":
                state.angle = float(val)
            elif key == "speedX":
                state.speed = float(val)
            elif key == "speedY":
                state.speedY = float(val)
            elif key == "speedZ":
                state.speedZ = float(val)
            elif key == "trackPos":
                state.trackPos = float(val)
            elif key == "track":
                state.track = _floats(val)
            elif key == "opponents":
                state.opponents = _floats(val)
            elif key == "rpm":
                state.rpm = float(val)
            elif key == "gear":
                state.gear = _int(val)
            elif key == "damage":
                state.damage = float(val)
            elif key == "distRaced":
                state.distRaced = float(val)
            elif key == "distFromStart":
                state.distFromStart = float(val)
            elif key == "lastLapTime":
                state.lastLapTime = float(val)
            elif key == "curLapTime":
                state.curLapTime = float(val)
            elif key == "racePos":
                state.racePos = _int(val)
            elif key == "fuel":
                state.fuel = float(val)
            elif key == "wheelSpinVel":
                state.wheelSpinVel = _floats(val)
            elif key == "z":
                state.z = float(val)

        return state
