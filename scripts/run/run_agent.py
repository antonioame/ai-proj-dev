"""Esegue il driver BC (_DRIVER/) contro un server TORCS.

Usage:
    python scripts/run/run_agent.py [--laps 1] [--host HOST] [--port PORT] [--telemetry]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _DRIVER.driver import BCDriver
from run_agent_common import run_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DRIVER_NAME = "bc"


def run(
    laps: int = 1,
    host: Optional[str] = None,
    port: Optional[int] = None,
    save_telemetry: bool = False,
) -> dict:
    return run_driver(BCDriver(), DRIVER_NAME, laps=laps, host=host, port=port, save_telemetry=save_telemetry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BC driver agent")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--telemetry", action="store_true",
                        help="Save full telemetry to data/<driver>_<ts>.csv")
    args = parser.parse_args()
    run(laps=args.laps, host=args.host, port=args.port, save_telemetry=args.telemetry)


if __name__ == "__main__":
    main()
