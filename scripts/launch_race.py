"""Launcher autonomo: avvia TORCS headless e poi esegue l'agente Python.

Usage:
    python scripts/launch_race.py [--laps 1] [--telemetry]

Lo script individua l'installazione di TORCS sotto U:\\AI-Partition\\torcs\\torcs,
avvia wtorcs.exe con -r (modalità gara), attende che il server apra la sua
porta UDP, poi lancia run_agent.py. Entrambi i processi vengono ripuliti
all'uscita.
"""

from __future__ import annotations

import argparse
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCS_EXE = Path(r"U:\AI-Partition\torcs\torcs\wtorcs.exe")
RACE_XML = PROJECT_ROOT / "torcs_env" / "race_config" / "corkscrew_solo.xml"
TORCS_HOST = "localhost"
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


def run_agent(laps: int, telemetry: bool) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_agent.py"),
        "--laps", str(laps),
    ]
    if telemetry:
        cmd.append("--telemetry")

    logger.info("Launching agent: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous TORCS race launcher (BC driver)")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()

    torcs_proc = start_torcs()

    try:
        logger.info("Waiting for TORCS to open UDP port %d (up to %ds)...", TORCS_PORT, TORCS_READY_TIMEOUT)
        if not wait_for_torcs(TORCS_PORT, TORCS_READY_TIMEOUT):
            logger.error("TORCS did not become ready in time. Aborting.")
            torcs_proc.terminate()
            sys.exit(1)
        logger.info("TORCS is ready.")

        exit_code = run_agent(args.laps, args.telemetry)
        logger.info("Agent finished with exit code %d.", exit_code)

    finally:
        if torcs_proc.poll() is None:
            logger.info("Terminating TORCS (PID %d).", torcs_proc.pid)
            torcs_proc.terminate()
            try:
                torcs_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                torcs_proc.kill()


if __name__ == "__main__":
    main()
