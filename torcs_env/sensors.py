"""Interpreta le stringhe di sensori SCR in una dataclass tipizzata.

Le stringhe di sensori SCR hanno questa forma:
  (angle 0.1)(speedX 50.2)(trackPos 0.0)(track 200 180 ...)(rpm 4500)...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Regex: intercetta i token (chiave val1 val2 ...)
_TOKEN_RE = re.compile(r'\((\w+)\s+([^)]+)\)')


def _floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split()]


def _int(raw: str) -> int:
    return int(float(raw))


@dataclass
class SensorState:
    # Orientamento auto rispetto all'asse pista (radianti, positivo = punta a sinistra)
    angle: float = 0.0

    # Velocità longitudinale / laterale / verticale (km/h)
    speed: float = 0.0
    speedY: float = 0.0
    speedZ: float = 0.0

    # Posizione in pista: 0 = centro, ±1 = bordo, > ±1 = fuori pista
    trackPos: float = 0.0

    # 19 letture dei range-finder (metri, max 200 m), da -45° a +45°, più fitti vicino a 0°
    track: list[float] = field(default_factory=lambda: [200.0] * 19)

    # 36 sensori di distanza dagli avversari (metri, max 200 m)
    opponents: list[float] = field(default_factory=lambda: [200.0] * 36)

    rpm: float = 0.0
    gear: int = 0
    damage: float = 0.0

    # Distanza percorsa dall'inizio gara (metri)
    distRaced: float = 0.0
    distFromStart: float = 0.0

    # Contatore giri derivato dai reset di distRaced (impostato dall'esterno dal client)
    lap: int = 1

    lastLapTime: float = 0.0
    curLapTime: float = 0.0
    racePos: int = 1
    fuel: float = 94.0

    # Velocità di rotazione delle quattro ruote (rad/s)
    wheelSpinVel: list[float] = field(default_factory=lambda: [0.0] * 4)

    # Altezza dell'auto rispetto alla superficie della pista (metri)
    z: float = 0.0

    # Stringa grezza (utile per il debug)
    raw: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_string(cls, sensor_str: str) -> "SensorState":
        """Interpreta una stringa di sensori SCR grezza in un SensorState."""
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
