"""Componenti condivisi da scripts/launch_race.py e scripts/launch_record.py:
avvio di TORCS headless, attesa dell'apertura della porta UDP, e lancio dello
script Python a valle come sottoprocesso.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCS_EXE = Path(os.environ.get("TORCS_EXE", r"U:\AI-Partition\torcs\torcs\wtorcs.exe"))
RACE_XML = PROJECT_ROOT / "torcs_env" / "race_config" / "corkscrew_solo.xml"
TORCS_PORT = 3001
TORCS_READY_TIMEOUT = 30   # secondi di attesa per l'apertura della porta da parte di TORCS
TORCS_POLL_INTERVAL = 0.5  # secondi tra un controllo di prontezza e l'altro


def _port_bound(port: int) -> bool:
    """Restituisce True se qualcosa è già collegato a *port* in UDP.

    Proviamo a fare il bind di un socket noi stessi — se fallisce, un altro
    processo lo possiede già. Inviare pacchetti probe SCR a TORCS
    corromperebbe il suo stato di handshake, quindi rileviamo la prontezza
    senza trasmettere alcun dato.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("", port))
        return False   # bind riuscito → la porta è libera
    except OSError:
        return True    # bind fallito → la porta è occupata (TORCS è attivo)
    finally:
        sock.close()


def wait_for_torcs(port: int = TORCS_PORT, timeout: float = TORCS_READY_TIMEOUT) -> bool:
    """Esegue polling finché TORCS non apre la porta UDP o scade *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_bound(port):
            return True
        time.sleep(TORCS_POLL_INTERVAL)
    return False


def start_torcs() -> subprocess.Popen:
    if not TORCS_EXE.exists():
        raise FileNotFoundError(f"TORCS executable not found: {TORCS_EXE}")
    if not RACE_XML.exists():
        raise FileNotFoundError(f"Race config not found: {RACE_XML}")

    logger.info("Starting TORCS: %s -r %s", TORCS_EXE, RACE_XML)
    proc = subprocess.Popen(
        [str(TORCS_EXE), "-r", str(RACE_XML)],
        cwd=str(TORCS_EXE.parent),
    )
    logger.info("TORCS started (PID %d)", proc.pid)
    return proc


def run_downstream_script(script_name: str, *extra_args: str) -> int:
    """Lancia uno script in scripts/ come sottoprocesso e ne restituisce il returncode."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), *extra_args]
    logger.info("Launching: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def stop_torcs(torcs_proc: subprocess.Popen) -> None:
    if torcs_proc.poll() is None:
        logger.info("Terminating TORCS (PID %d).", torcs_proc.pid)
        torcs_proc.terminate()
        try:
            torcs_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            torcs_proc.kill()
