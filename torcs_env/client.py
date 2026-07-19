"""Client UDP che implementa il protocollo SCR (Simulated Car Racing) per TORCS.

Panoramica del protocollo
--------------------------
1. Alla prima connessione il client invia una stringa di identificazione:
     SCR(init -45 -38 -30 -22 -15 -10 -6 -3 -1 0 1 3 6 10 15 22 30 38 45)
   I numeri sono gli angoli dei rangefinder in gradi.

2. Il server risponde con una stringa di sensori ad ogni step di simulazione (20 ms circa, 50 step/s).

3. Il client risponde con una stringa di controllo:
     (accel X)(brake X)(steer X)(gear X)(clutch X)(meta X)

4. Il server invia "***restart***" per segnalare il restart della gara.
   Il server invia "***shutdown***" per segnalare che si sta chiudendo.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Optional

from .actions import Action
from .sensors import SensorState

logger = logging.getLogger(__name__)

# Angoli di default dei sensori per i 19 rangefinder (gradi)
_DEFAULT_ANGLES = [-45, -38, -30, -22, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15, 22, 30, 38, 45]

_MSG_RESTART = b"***restart***"
_MSG_SHUTDOWN = b"***shutdown***"
_MSG_IDENTIFIED = b"***identified***"

# Valori sentinella restituiti da receive() per segnalare eventi di protocollo
RESTART = "RESTART"
SHUTDOWN = "SHUTDOWN"


class TORCSClient:
    """Client UDP per il server SCR di TORCS."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        sensor_angles: Optional[list[int]] = None,
        timeout: float = 30.0,
        max_reconnect_attempts: int = 10,
    ) -> None:
        self.host = host or os.environ.get("TORCS_HOST", "localhost")
        self.port = int(port or os.environ.get("TORCS_PORT", "3001"))
        # Evita il default mutabile condiviso fra istanze: copia sempre una nuova lista
        self.sensor_angles = list(sensor_angles) if sensor_angles is not None else list(_DEFAULT_ANGLES)
        self.timeout = timeout
        self.max_reconnect_attempts = max_reconnect_attempts

        self._sock: Optional[socket.socket] = None
        self._server_addr = (self.host, self.port)

        # Conteggio giri tramite i reset di distRaced
        self._prev_dist_raced: float = 0.0
        self._lap: int = 1

    # ------------------------------------------------------------------
    # Gestione della connessione
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Crea il socket UDP ed esegue l'handshake SCR."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout)
        self._handshake()

    def _handshake(self) -> None:
        """Invia la stringa di identificazione SCR e attende la prima risposta del server."""
        angles_str = " ".join(str(a) for a in self.sensor_angles)
        init_msg = f"SCR(init {angles_str})".encode()

        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                logger.info(
                    "Connecting to TORCS at %s:%d (attempt %d)",
                    self.host, self.port, attempt,
                )
                self._sock.sendto(init_msg, self._server_addr)
                data, _ = self._sock.recvfrom(4096)
                # Rimuove i terminatori null che il server SCR aggiunge
                clean = data.rstrip(b'\x00')
                if clean == _MSG_IDENTIFIED or clean.startswith(_MSG_IDENTIFIED):
                    logger.info("Handshake successful (server identified client).")
                    return
                if clean == _MSG_RESTART or clean == _MSG_SHUTDOWN:
                    logger.warning("Received control message during handshake: %s", clean)
                    return
                logger.info("Handshake successful.")
                return
            except (socket.timeout, ConnectionResetError):
                # Su Windows recvfrom() può sollevare ConnectionResetError (WinError 10054,
                # ICMP port unreachable) se TORCS non ha ancora aperto la porta SCR quando
                # arriva il primo pacchetto: va trattato come un normale timeout da ritentare.
                wait = 2 ** attempt
                logger.warning("Handshake timeout. Retrying in %d s…", wait)
                time.sleep(wait)

        raise ConnectionError(
            f"Could not connect to TORCS at {self.host}:{self.port} "
            f"after {self.max_reconnect_attempts} attempts."
        )

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------------
    # Comunicazione
    # ------------------------------------------------------------------

    def receive(self) -> SensorState | str:
        """Riceve un pacchetto di sensori dal server.

        Returns
        -------
        SensorState
            Dati dei sensori interpretati per questo step di simulazione.
        str
            Una delle sentinelle RESTART o SHUTDOWN.
        """
        assert self._sock is not None, "Call connect() first."

        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                data, _ = self._sock.recvfrom(4096)
                break
            except ConnectionResetError as exc:
                # Windows solleva WinError 10054 quando TORCS chiude la porta UDP
                # (ICMP Port Unreachable). Di solito significa che TORCS ha interrotto
                # la gara perché il timeout SCR di pre-connessione è scattato prima
                # dell'handshake, oppure il timeout per-azione è scattato a gara in corso.
                raise ConnectionError(
                    "TORCS reset the connection (WinError 10054). "
                    "The SCR server likely timed out waiting for the client. "
                    "Ensure the driver connects and sends the first action before "
                    "TORCS's SCR timeouts fire (about 2-3 s pre-connection, about 2.85 s per-action)."
                ) from exc
            except socket.timeout:
                if attempt == self.max_reconnect_attempts:
                    raise TimeoutError(
                        f"No data from TORCS after {self.max_reconnect_attempts} attempts."
                    )
                wait = min(2 ** attempt, 16)
                logger.warning("Receive timeout (attempt %d). Retrying in %d s…", attempt, wait)
                time.sleep(wait)

        clean = data.rstrip(b'\x00')
        if clean == _MSG_RESTART:
            self._lap = 1
            self._prev_dist_raced = 0.0
            return RESTART
        if clean == _MSG_SHUTDOWN:
            return SHUTDOWN

        sensor_str = clean.decode(errors="replace")
        state = SensorState.from_string(sensor_str)
        state.lap = self._update_lap(state.distRaced)
        return state

    def send(self, action: Action) -> None:
        """Invia un'azione di controllo al server."""
        assert self._sock is not None, "Call connect() first."
        msg = action.clamp().to_string().encode()
        self._sock.sendto(msg, self._server_addr)

    def send_restart(self) -> None:
        """Chiede al server di riavviare la gara."""
        assert self._sock is not None, "Call connect() first."
        self._sock.sendto(b"(meta 1)", self._server_addr)

    def send_shutdown(self) -> None:
        """Segnala al server di terminare la sessione in modo pulito (meta 2)."""
        assert self._sock is not None, "Call connect() first."
        msg = Action(meta=2).to_string().encode()
        self._sock.sendto(msg, self._server_addr)

    # ------------------------------------------------------------------
    # Conteggio giri
    # ------------------------------------------------------------------

    def _update_lap(self, dist_raced: float) -> int:
        """Incrementa il contatore giri quando distRaced si resetta (inizia un nuovo giro)."""
        if dist_raced < self._prev_dist_raced - 100.0:
            # distRaced è calato: il server lo ha resettato per il nuovo giro
            self._lap += 1
        self._prev_dist_raced = dist_raced
        return self._lap

    # ------------------------------------------------------------------
    # Gestore di contesto
    # ------------------------------------------------------------------

    def __enter__(self) -> "TORCSClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()
