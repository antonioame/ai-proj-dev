"""Valuta un checkpoint CEM (drivers/cem/) e salva un JSON di risultati
strutturato, stesso schema di scripts/eval/evaluate.py/evaluate_rl.py.

Usage:
    python scripts/eval/evaluate_cem.py [--laps 1] [--checkpoint drivers/rl/models/cem_v1.pth]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drivers.cem.driver import CemDriver
from evaluate_common import run_eval_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def evaluate(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: Path | None = None,
    max_steps: int = 20000,
    checkpoint: str | None = None,
) -> dict:
    driver = CemDriver(checkpoint_path=Path(checkpoint)) if checkpoint else CemDriver()
    return run_eval_loop(
        driver, "cem", laps=laps, host=host, port=port,
        output_path=output_path, max_steps=max_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CEM TORCS driver")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    evaluate(
        laps=args.laps,
        host=args.host,
        port=args.port,
        output_path=Path(args.output) if args.output else None,
        max_steps=args.max_steps,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
