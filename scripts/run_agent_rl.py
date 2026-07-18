"""Esegue il driver RL di Fase 3 (drivers/rl/) contro un server TORCS.

Entry point dedicato — rispecchia esattamente scripts/run_agent.py ma
codifica RLDriver invece di BCDriver, allo stesso modo in cui
old_versions_drivers/project_V2/run_rule_based.py è un entry point dedicato
per il driver rule_based. run_agent.py stesso resta intatto, così BC rimane
il fallback a rischio zero.

Usage:
    python scripts/run_agent_rl.py [--laps 1] [--host HOST] [--port PORT] [--telemetry]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drivers.rl.driver import RLDriver
from run_agent_common import run_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DRIVER_NAME = "rl"


def run(
    laps: int = 1,
    host: Optional[str] = None,
    port: Optional[int] = None,
    save_telemetry: bool = False,
    residual: bool = False,
    checkpoint: Optional[str] = None,
) -> dict:
    if residual:
        from drivers.rl.residual_driver import ResidualRLDriver
        driver_name = "rl_residual"
        driver = ResidualRLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else ResidualRLDriver()
    else:
        driver_name = DRIVER_NAME
        driver = RLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else RLDriver()

    return run_driver(driver, driver_name, laps=laps, host=host, port=port, save_telemetry=save_telemetry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 RL driver agent")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--telemetry", action="store_true",
                        help="Save full telemetry to data/<driver>_<ts>.csv")
    parser.add_argument("--residual", action="store_true",
                        help="Run the residual driver (BC base + SAC correction).")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a SAC .zip checkpoint (default depends on --residual).")
    args = parser.parse_args()
    run(laps=args.laps, host=args.host, port=args.port, save_telemetry=args.telemetry,
        residual=args.residual, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
