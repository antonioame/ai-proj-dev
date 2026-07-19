"""Launcher autonomo: avvia TORCS headless e poi esegue lo script a valle.

Usage:
    python scripts/launch/launch_race.py [--laps 1] [--telemetry]
    python scripts/launch/launch_race.py --mode record [--laps 3]

Lo script individua l'installazione di TORCS sotto U:\\AI-Partition\\torcs\\torcs,
avvia wtorcs.exe con -r (modalità gara), attende che il server apra la sua
porta UDP, poi lancia run_agent.py (--mode race, default) o record_agent.py
(--mode record). Entrambi i processi vengono ripuliti all'uscita.
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
    parser = argparse.ArgumentParser(description="Autonomous TORCS launcher (race or record mode)")
    parser.add_argument("--mode", choices=["race", "record"], default="race")
    parser.add_argument("--laps", type=int, default=None,
                        help="Defaults to 1 for --mode race, 3 for --mode record")
    parser.add_argument("--telemetry", action="store_true", help="Only for --mode race")
    args = parser.parse_args()
    laps = args.laps if args.laps is not None else (1 if args.mode == "race" else 3)

    torcs_proc = start_torcs()

    try:
        logger.info("Waiting for TORCS to open UDP port %d (up to %ds)...", TORCS_PORT, TORCS_READY_TIMEOUT)
        if not wait_for_torcs(TORCS_PORT, TORCS_READY_TIMEOUT):
            logger.error("TORCS did not become ready in time. Aborting.")
            torcs_proc.terminate()
            sys.exit(1)
        logger.info("TORCS is ready.")

        if args.mode == "race":
            extra_args = ["--laps", str(laps)]
            if args.telemetry:
                extra_args.append("--telemetry")
            exit_code = run_downstream_script("run/run_agent.py", *extra_args)
        else:
            exit_code = run_downstream_script("record/record_agent.py", "--laps", str(laps))
        logger.info("Downstream script finished with exit code %d.", exit_code)

    finally:
        stop_torcs(torcs_proc)


if __name__ == "__main__":
    main()
