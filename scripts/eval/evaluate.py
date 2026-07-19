"""Valuta il driver BC (_DRIVER/) e salva un JSON di risultati strutturato.

Usage:
    python scripts/eval/evaluate.py [--laps 1] [--output results/eval.json]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _DRIVER.driver import BCDriver
from evaluate_common import run_eval_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def evaluate(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: Path | None = None,
) -> dict:
    return run_eval_loop(BCDriver(), "bc", laps=laps, host=host, port=port, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the BC TORCS driver")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    evaluate(
        laps=args.laps,
        host=args.host,
        port=args.port,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
