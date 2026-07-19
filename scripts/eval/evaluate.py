"""Valuta un driver TORCS (bc, cem, bc_dagger, rl) e salva un JSON di risultati
strutturato, stesso schema per tutti i driver così i risultati sono
direttamente confrontabili.

Usage:
    python scripts/eval/evaluate.py --driver bc [--laps 1] [--output results/eval.json]
    python scripts/eval/evaluate.py --driver cem [--checkpoint drivers/rl/models/cem_v1.pth]
    python scripts/eval/evaluate.py --driver bc_dagger [--checkpoint-name bc_dagger_v1]
    python scripts/eval/evaluate.py --driver rl [--residual] [--checkpoint path/to.zip]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_common import run_eval_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "_DRIVER" / "models"


def _build_bc():
    from _DRIVER.driver import BCDriver

    return BCDriver(), "bc", {}


def _build_cem(checkpoint: str | None):
    from drivers.cem.driver import CemDriver

    driver = CemDriver(checkpoint_path=Path(checkpoint)) if checkpoint else CemDriver()
    return driver, "cem", {"max_steps": 20000}


def _build_bc_dagger(checkpoint_name: str):
    from drivers.bc_dagger.driver import BCDaggerDriver

    driver = BCDaggerDriver(
        checkpoint_path=MODELS_DIR / f"{checkpoint_name}.pth",
        stats_path=MODELS_DIR / f"{checkpoint_name}.npz",
    )
    return driver, checkpoint_name, {}


def _build_rl(checkpoint: str | None, residual: bool):
    if residual:
        from drivers.rl.residual_driver import ResidualRLDriver

        driver = ResidualRLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else ResidualRLDriver()
        return driver, "rl_residual", {"max_steps": 20000}

    from drivers.rl.driver import RLDriver

    driver = RLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else RLDriver()
    return driver, "rl", {"max_steps": 20000}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TORCS driver (bc, cem, bc_dagger, rl)")
    parser.add_argument("--driver", choices=["bc", "cem", "bc_dagger", "rl"], default="bc")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path, used by --driver cem/rl")
    parser.add_argument("--checkpoint-name", default="bc_dagger_v1",
                        help="Checkpoint name (without extension), used by --driver bc_dagger")
    parser.add_argument("--residual", action="store_true",
                        help="With --driver rl: evaluate the residual driver (BC base + SAC correction)")
    args = parser.parse_args()

    if args.driver == "bc":
        driver, driver_name, extra = _build_bc()
    elif args.driver == "cem":
        driver, driver_name, extra = _build_cem(args.checkpoint)
    elif args.driver == "bc_dagger":
        driver, driver_name, extra = _build_bc_dagger(args.checkpoint_name)
    else:
        driver, driver_name, extra = _build_rl(args.checkpoint, args.residual)

    run_eval_loop(
        driver, driver_name, laps=args.laps, host=args.host, port=args.port,
        output_path=Path(args.output) if args.output else None, **extra,
    )


if __name__ == "__main__":
    main()
