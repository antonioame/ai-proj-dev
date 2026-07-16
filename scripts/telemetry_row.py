"""Costruzione condivisa della riga di telemetria CSV usata da record_agent.py,
record_human.py e record_keyboard.py: stesso set base di colonne sensori/azioni,
con distFromStart opzionale (solo record_agent.py lo include).
"""

from __future__ import annotations

from torcs_env.actions import Action
from torcs_env.sensors import SensorState

TRACK_COLS = [f"track_{i}" for i in range(19)]


def build_row(timestamp: float, state: SensorState, action: Action, include_dist_from_start: bool = False) -> dict:
    row = {
        "timestamp": timestamp,
        "angle": state.angle,
        "speed": state.speed,
        "speedY": state.speedY,
        "speedZ": state.speedZ,
        "trackPos": state.trackPos,
        **{f"track_{i}": state.track[i] for i in range(min(19, len(state.track)))},
        "rpm": state.rpm,
        "gear": state.gear,
    }
    if include_dist_from_start:
        row["distFromStart"] = state.distFromStart
    row.update({
        "distRaced": state.distRaced,
        "curLapTime": state.curLapTime,
        "steer": action.steer,
        "accel": action.accel,
        "brake": action.brake,
        "gear_cmd": action.gear,
    })
    return row
