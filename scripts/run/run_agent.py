"""Esegue un driver TORCS (bc, rl) contro un server TORCS.

Usage:
    python scripts/run/run_agent.py --driver bc [--laps 1] [--host HOST] [--port PORT] [--telemetry]
    python scripts/run/run_agent.py --driver rl [--residual] [--checkpoint path/to.zip]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_agent_common import run_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _build_bc():
    from _DRIVER.driver import BCDriver

    return BCDriver(), "bc"


def _build_rl(checkpoint: Optional[str], residual: bool):
    if residual:
        from drivers.rl.residual_driver import ResidualRLDriver

        driver = ResidualRLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else ResidualRLDriver()
        return driver, "rl_residual"

    from drivers.rl.driver import RLDriver

    driver = RLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else RLDriver()
    return driver, "rl"


def run(
    laps: int = 1,
    host: Optional[str] = None,
    port: Optional[int] = None,
    save_telemetry: bool = False,
    driver_kind: str = "bc",
    checkpoint: Optional[str] = None,
    residual: bool = False,
) -> dict:
    driver, driver_name = _build_bc() if driver_kind == "bc" else _build_rl(checkpoint, residual)
    return run_driver(driver, driver_name, laps=laps, host=host, port=port, save_telemetry=save_telemetry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TORCS driver agent (bc, rl)")
    parser.add_argument("--driver", choices=["bc", "rl"], default="bc")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--telemetry", action="store_true",
                        help="Save full telemetry to data/<driver>_<ts>.csv")
    parser.add_argument("--residual", action="store_true",
                        help="With --driver rl: run the residual driver (BC base + SAC correction)")
    parser.add_argument("--checkpoint", default=None,
                        help="With --driver rl: path to a SAC .zip checkpoint (default depends on --residual)")
    args = parser.parse_args()
    run(
        laps=args.laps, host=args.host, port=args.port, save_telemetry=args.telemetry,
        driver_kind=args.driver, checkpoint=args.checkpoint, residual=args.residual,
    )


if __name__ == "__main__":
    main()
