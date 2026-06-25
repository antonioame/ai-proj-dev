"""Abstract base class all drivers must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torcs_env.actions import Action
from torcs_env.sensors import SensorState


class BaseDriver(ABC):
    """Interface for any TORCS driving agent.

    Subclasses implement `step()` to map a SensorState to an Action.
    Optional lifecycle hooks: `on_restart()` and `on_shutdown()`.
    """

    @abstractmethod
    def step(self, state: SensorState) -> Action:
        """Return the action to take given the current sensor state."""

    def on_restart(self) -> None:
        """Called when the TORCS server signals a race restart."""

    def on_shutdown(self) -> None:
        """Called when the TORCS server signals it is shutting down."""

    def reset(self) -> None:
        """Reset any internal state (e.g., integrators, stuck timers)."""
