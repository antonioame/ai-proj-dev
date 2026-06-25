"""Autonomous launcher: starts TORCS headless then runs the Python agent.

Usage:
    python scripts/launch_race.py [--driver rule_based] [--laps 1] [--telemetry]

The script discovers the TORCS installation under U:\\AI-Partition\\torcs\\torcs,
starts wtorcs.exe with -r (race mode), waits for the server to open its UDP
port, then launches run_agent.py.  Both processes are cleaned up on exit.
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
TORCS_READY_TIMEOUT = 30   # seconds to wait for TORCS to open the port
TORCS_POLL_INTERVAL = 0.5  # seconds between readiness checks


def _port_bound(port: int) -> bool:
    """Return True if something is already bound to *port* on UDP.

    We try to bind a socket ourselves — if it fails, another process owns it.
    Sending SCR probe packets to TORCS would corrupt its handshake state, so
    we detect readiness without transmitting any data.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("", port))
        return False   # bind succeeded → port is free
    except OSError:
        return True    # bind failed → port is taken (TORCS is up)
    finally:
        sock.close()


def wait_for_torcs(port: int = TORCS_PORT, timeout: float = TORCS_READY_TIMEOUT) -> bool:
    """Poll until TORCS binds the UDP port or *timeout* elapses."""
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


def run_agent(driver: str, laps: int, telemetry: bool) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_agent.py"),
        "--driver", driver,
        "--laps", str(laps),
    ]
    if telemetry:
        cmd.append("--telemetry")

    logger.info("Launching agent: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous TORCS race launcher")
    parser.add_argument("--driver", default="rule_based")
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

        exit_code = run_agent(args.driver, args.laps, args.telemetry)
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
