"""Launcher autonomo per la registrazione: avvia TORCS headless e poi esegue record_agent.py.

Usage:
    python scripts/launch_record.py [--laps 3]

Script di supporto temporaneo per la raccolta dati di riaddestramento —
rispecchia launch_race.py ma chiama record_agent.py invece di run_agent.py.
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
TORCS_PORT = 3001
TORCS_READY_TIMEOUT = 30
TORCS_POLL_INTERVAL = 0.5


def _port_bound(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("", port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def wait_for_torcs(port: int = TORCS_PORT, timeout: float = TORCS_READY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_bound(port):
            return True
        time.sleep(TORCS_POLL_INTERVAL)
    return False


def start_torcs() -> subprocess.Popen:
    logger.info("Starting TORCS: %s -r %s", TORCS_EXE, RACE_XML)
    proc = subprocess.Popen(
        [str(TORCS_EXE), "-r", str(RACE_XML)],
        cwd=str(TORCS_EXE.parent),
    )
    logger.info("TORCS started (PID %d)", proc.pid)
    return proc


def run_record(laps: int) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "record_agent.py"), "--laps", str(laps)]
    logger.info("Launching recorder: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laps", type=int, default=3)
    args = parser.parse_args()

    torcs_proc = start_torcs()
    try:
        logger.info("Waiting for TORCS to open UDP port %d...", TORCS_PORT)
        if not wait_for_torcs(TORCS_PORT, TORCS_READY_TIMEOUT):
            logger.error("TORCS did not become ready in time. Aborting.")
            torcs_proc.terminate()
            sys.exit(1)
        logger.info("TORCS is ready.")
        exit_code = run_record(args.laps)
        logger.info("Recorder finished with exit code %d.", exit_code)
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
