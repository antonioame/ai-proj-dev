"""UDP client implementing the SCR (Simulated Car Racing) protocol for TORCS.

Protocol overview
-----------------
1. Client sends an identification string on first connect:
     SCR(init -45 -38 -30 -22 -15 -10 -6 -3 -1 0 1 3 6 10 15 22 30 38 45)
   The numbers are the rangefinder angles in degrees.

2. Server responds with a sensor string each simulation step (~50 ms).

3. Client responds with a control string:
     (accel X)(brake X)(steer X)(gear X)(clutch X)(meta X)

4. Server sends "***restart***" to signal a race restart.
   Server sends "***shutdown***" to signal it is closing.
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

# Default sensor angles for the 19 rangefinders (degrees)
_DEFAULT_ANGLES = [-45, -38, -30, -22, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15, 22, 30, 38, 45]

_MSG_RESTART = b"***restart***"
_MSG_SHUTDOWN = b"***shutdown***"
_MSG_IDENTIFIED = b"***identified***"

# Sentinel values returned by receive() to signal protocol events
RESTART = "RESTART"
SHUTDOWN = "SHUTDOWN"


class TORCSClient:
    """UDP client for the SCR TORCS server."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        sensor_angles: list[int] = _DEFAULT_ANGLES,
        timeout: float = 10.0,
        max_reconnect_attempts: int = 5,
    ) -> None:
        self.host = host or os.environ.get("TORCS_HOST", "localhost")
        self.port = int(port or os.environ.get("TORCS_PORT", "3001"))
        self.sensor_angles = sensor_angles
        self.timeout = timeout
        self.max_reconnect_attempts = max_reconnect_attempts

        self._sock: Optional[socket.socket] = None
        self._server_addr = (self.host, self.port)

        # Lap tracking via distRaced resets
        self._prev_dist_raced: float = 0.0
        self._lap: int = 1

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create UDP socket and perform the SCR handshake."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout)
        self._handshake()

    def _handshake(self) -> None:
        """Send the SCR identification string and wait for the first sensor."""
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
                # Strip null terminators the SCR server appends
                clean = data.rstrip(b'\x00')
                if clean == _MSG_IDENTIFIED or clean.startswith(_MSG_IDENTIFIED):
                    logger.info("Handshake successful (server identified client).")
                    return
                if clean == _MSG_RESTART or clean == _MSG_SHUTDOWN:
                    logger.warning("Received control message during handshake: %s", clean)
                    return
                logger.info("Handshake successful.")
                return
            except socket.timeout:
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
    # Communication
    # ------------------------------------------------------------------

    def receive(self) -> SensorState | str:
        """Receive one sensor packet from the server.

        Returns
        -------
        SensorState
            Parsed sensor data for this simulation step.
        str
            One of the RESTART or SHUTDOWN sentinels.
        """
        assert self._sock is not None, "Call connect() first."

        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                data, _ = self._sock.recvfrom(4096)
                break
            except socket.timeout:
                if attempt == self.max_reconnect_attempts:
                    raise TimeoutError(
                        f"No data from TORCS after {self.max_reconnect_attempts} attempts."
                    )
                wait = min(2 ** attempt, 16)
                logger.warning("Receive timeout (attempt %d). Retrying in %d s…", attempt, wait)
                time.sleep(wait)

        if data == _MSG_RESTART:
            self._lap = 1
            self._prev_dist_raced = 0.0
            return RESTART
        if data == _MSG_SHUTDOWN:
            return SHUTDOWN

        sensor_str = data.decode(errors="replace")
        state = SensorState.from_string(sensor_str)
        state.lap = self._update_lap(state.distRaced)
        return state

    def send(self, action: Action) -> None:
        """Send a control action to the server."""
        assert self._sock is not None, "Call connect() first."
        msg = action.clamp().to_string().encode()
        self._sock.sendto(msg, self._server_addr)

    def send_restart(self) -> None:
        """Ask the server to restart the race."""
        assert self._sock is not None, "Call connect() first."
        self._sock.sendto(b"(meta 1)", self._server_addr)

    def send_shutdown(self) -> None:
        """Signal the server to end the session cleanly (meta 2)."""
        assert self._sock is not None, "Call connect() first."
        msg = Action(meta=2).to_string().encode()
        self._sock.sendto(msg, self._server_addr)

    # ------------------------------------------------------------------
    # Lap tracking
    # ------------------------------------------------------------------

    def _update_lap(self, dist_raced: float) -> int:
        """Increment lap counter when distRaced resets (new lap starts)."""
        if dist_raced < self._prev_dist_raced - 100.0:
            # distRaced dropped — server reset it for the new lap
            self._lap += 1
        self._prev_dist_raced = dist_raced
        return self._lap

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TORCSClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()
