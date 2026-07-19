"""Valuta il driver bc_dagger_v1 (drivers/bc_dagger/) e salva un JSON di
risultati strutturato.

Entry point dedicato — rispecchia esattamente scripts/eval/evaluate.py ma
codifica BCDaggerDriver invece di BCDriver, così evaluate.py stesso resta
intatto e la baseline BC che valuta rimane il fallback a rischio zero.
Produce lo stesso schema JSON di evaluate.py/evaluate_rl.py (tempo giro,
frazione fuori pista, danni) così i risultati sono direttamente
confrontabili fianco a fianco.

Usage:
    python scripts/eval/evaluate_bc_dagger.py [--laps 1] [--output results/eval_bc_dagger.json]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drivers.bc_dagger.driver import BCDaggerDriver
from evaluate_common import run_eval_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def evaluate(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: Path | None = None,
    checkpoint_name: str = "bc_dagger_v1",
) -> dict:
    driver_name = checkpoint_name
    models_dir = Path(__file__).resolve().parent.parent.parent / "_DRIVER" / "models"
    driver = BCDaggerDriver(
        checkpoint_path=models_dir / f"{checkpoint_name}.pth",
        stats_path=models_dir / f"{checkpoint_name}.npz",
    )
    return run_eval_loop(driver, driver_name, laps=laps, host=host, port=port, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the bc_dagger_v1 TORCS driver")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-name", default="bc_dagger_v1")
    args = parser.parse_args()

    evaluate(
        laps=args.laps,
        host=args.host,
        port=args.port,
        output_path=Path(args.output) if args.output else None,
        checkpoint_name=args.checkpoint_name,
    )


if __name__ == "__main__":
    main()
