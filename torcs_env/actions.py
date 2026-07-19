"""Dataclass Action: incapsula la stringa di controllo SCR inviata al server TORCS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Action:
    # Sterzo: -1 (tutto a destra) a +1 (tutto a sinistra)
    steer: float = 0.0

    # Acceleratore: 0-1
    accel: float = 0.0

    # Freno: 0-1
    brake: float = 0.0

    # Marcia: -1 (retromarcia), 0 (folle), 1-6
    gear: int = 1

    # Frizione: 0-1
    clutch: float = 0.0

    # Meta opzionale: richiede il restart della gara o lo shutdown del client
    meta: int = 0

    def to_string(self) -> str:
        """Serializza nel formato della stringa di controllo SCR."""
        return (
            f"(accel {self.accel:.4f})"
            f"(brake {self.brake:.4f})"
            f"(steer {self.steer:.4f})"
            f"(gear {self.gear})"
            f"(clutch {self.clutch:.4f})"
            f"(meta {self.meta})"
        )

    def clamp(self) -> "Action":
        """Restituisce una nuova Action con tutti i valori limitati agli intervalli validi."""
        return Action(
            steer=max(-1.0, min(1.0, self.steer)),
            accel=max(0.0, min(1.0, self.accel)),
            brake=max(0.0, min(1.0, self.brake)),
            gear=max(-1, min(6, self.gear)),
            clutch=max(0.0, min(1.0, self.clutch)),
            meta=self.meta,
        )
