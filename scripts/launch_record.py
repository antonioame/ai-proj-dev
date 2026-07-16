"""Launcher autonomo per la registrazione: avvia TORCS headless e poi esegue record_agent.py.

Usage:
    python scripts/launch_record.py [--laps 3]

Script di supporto temporaneo per la raccolta dati di riaddestramento —
rispecchia launch_race.py ma chiama record_agent.py invece di run_agent.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch_common import TORCS_PORT, TORCS_READY_TIMEOUT, run_downstream_script, start_torcs, stop_torcs, wait_for_torcs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
        exit_code = run_downstream_script("record_agent.py", "--laps", str(args.laps))
        logger.info("Recorder finished with exit code %d.", exit_code)
    finally:
        stop_torcs(torcs_proc)


if __name__ == "__main__":
    main()
