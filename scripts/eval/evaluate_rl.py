"""Valuta il driver RL di Fase 3 (drivers/rl/) e salva un JSON di risultati strutturato.

Entry point dedicato — rispecchia esattamente scripts/eval/evaluate.py ma
codifica RLDriver invece di BCDriver, così evaluate.py stesso resta intatto
e la baseline BC che valuta rimane il fallback a rischio zero. Produce lo
stesso schema JSON di evaluate.py (tempo giro, frazione fuori pista, danni)
così i risultati sono direttamente confrontabili fianco a fianco.

Usage:
    python scripts/eval/evaluate_rl.py [--laps 1] [--output results/eval_rl.json]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drivers.rl.driver import RLDriver
from evaluate_common import run_eval_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def evaluate(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: Path | None = None,
    max_steps: int = 20000,
    checkpoint: str | None = None,
    residual: bool = False,
) -> dict:
    if residual:
        from drivers.rl.residual_driver import ResidualRLDriver
        driver_name = "rl_residual"
        driver = ResidualRLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else ResidualRLDriver()
    else:
        driver_name = "rl"
        driver = RLDriver(checkpoint_path=Path(checkpoint)) if checkpoint else RLDriver()

    return run_eval_loop(
        driver, driver_name, laps=laps, host=host, port=port,
        output_path=output_path, max_steps=max_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 3 RL TORCS driver")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=20000,
                        help="Abort (record no-lap) if a lap isn't completed within this many steps.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a SAC .zip checkpoint (default depends on --residual).")
    parser.add_argument("--residual", action="store_true",
                        help="Evaluate the residual driver (BC base + SAC correction).")
    args = parser.parse_args()

    evaluate(
        laps=args.laps,
        host=args.host,
        port=args.port,
        output_path=Path(args.output) if args.output else None,
        max_steps=args.max_steps,
        checkpoint=args.checkpoint,
        residual=args.residual,
    )


if __name__ == "__main__":
    main()
