"""Classe base astratta che tutti i driver devono implementare."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torcs_env.actions import Action
from torcs_env.sensors import SensorState


class BaseDriver(ABC):
    """Interfaccia per qualsiasi agente di guida TORCS.

    Le sottoclassi implementano `step()` per mappare uno SensorState a un Action.
    Hook del ciclo di vita opzionali: `on_restart()` e `on_shutdown()`.
    """

    @abstractmethod
    def step(self, state: SensorState) -> Action:
        """Restituisci l'azione da eseguire dato lo stato dei sensori attuali."""

    def on_restart(self) -> None:
        """Chiamato quando il server TORCS segnala un riavvio di gara."""

    def on_shutdown(self) -> None:
        """Chiamato quando il server TORCS segnala che sta per spegnersi."""

    def reset(self) -> None:
        """Ripristina qualsiasi stato interno (es. integratori, timer stuck)."""
