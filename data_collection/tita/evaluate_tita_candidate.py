"""Valuta il driver candidato bc_tita_v1 (data_collection/tita/driver_candidate.py)
e salva un JSON di risultati strutturato, stesso schema di scripts/eval/evaluate.py,
cosi' il confronto con bc (121.978 s) e' diretto.

Non tocca _DRIVER/, drivers/, scripts/, results/: legge il checkpoint da
data_collection/tita/candidate_models/ e scrive l'esito in
data_collection/tita/results/.

Prerequisito: TORCS gia' avviato (wtorcs.exe -r <config>), stesso schema di
connessione di scripts/eval/evaluate.py.

Usage:
    python data_collection/tita/evaluate_tita_candidate.py --laps 1
    python data_collection/tita/evaluate_tita_candidate.py --laps 5 --output data_collection/tita/results/eval_bc_tita_v1.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "eval"))

from data_collection.tita.driver_candidate import BCTitaCandidateDriver
from evaluate_common import run_eval_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CANDIDATE_DIR = Path(__file__).resolve().parent / "candidate_models"


def evaluate(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: Path | None = None,
    checkpoint_name: str = "bc_tita_v1",
) -> dict:
    driver = BCTitaCandidateDriver(
        checkpoint_path=CANDIDATE_DIR / f"{checkpoint_name}.pth",
        stats_path=CANDIDATE_DIR / f"{checkpoint_name}.npz",
    )
    if output_path is None:
        RESULTS_DIR.mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"eval_{checkpoint_name}_{date_str}.json"
    return run_eval_loop(driver, checkpoint_name, laps=laps, host=host, port=port, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the bc_tita_v1 candidate TORCS driver")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-name", default="bc_tita_v1")
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
