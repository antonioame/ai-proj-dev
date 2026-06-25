"""Action dataclass: wraps the SCR control string sent to the TORCS server."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Action:
    # Steering: -1 (full right) to +1 (full left)
    steer: float = 0.0

    # Throttle: 0–1
    accel: float = 0.0

    # Brake: 0–1
    brake: float = 0.0

    # Gear: -1 (reverse), 0 (neutral), 1–6
    gear: int = 1

    # Clutch: 0–1
    clutch: float = 0.0

    # Optional meta: request race restart or client shutdown
    meta: int = 0

    def to_string(self) -> str:
        """Serialise to the SCR control string format."""
        return (
            f"(accel {self.accel:.4f})"
            f"(brake {self.brake:.4f})"
            f"(steer {self.steer:.4f})"
            f"(gear {self.gear})"
            f"(clutch {self.clutch:.4f})"
            f"(meta {self.meta})"
        )

    def clamp(self) -> "Action":
        """Return a new Action with all values clamped to valid ranges."""
        return Action(
            steer=max(-1.0, min(1.0, self.steer)),
            accel=max(0.0, min(1.0, self.accel)),
            brake=max(0.0, min(1.0, self.brake)),
            gear=max(-1, min(6, self.gear)),
            clutch=max(0.0, min(1.0, self.clutch)),
            meta=self.meta,
        )
