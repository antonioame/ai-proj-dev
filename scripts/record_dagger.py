"""Raccolta dati DAgger: fa girare una policy esistente (BC o residual SAC) su
TORCS e, ad ogni tick, interroga in ombra il vecchio RuleBasedDriver
(old_versions_drivers/project_V2) come oracolo per etichettare lo stato
visitato con l'azione "corretta" secondo l'oracolo.

L'oracolo NON guida l'auto — la sua azione viene solo calcolata e loggata
per ogni stato realmente visitato dalla policy in rollout (questo è il punto
del DAgger: correggere gli stati fuori distribuzione che la policy visita
davvero, non solo quelli dell'oracolo).

Output: CSV con le 26 feature di build_feature_vector() + le 3 azioni
dell'oracolo (steer/accel/brake), una riga per tick a 50Hz.

Vincoli: script nuovo e separato, non tocca _DRIVER/driver.py né i file di
Fase 1/2/3 esistenti.

Usage:
    python scripts/record_dagger.py --policy bc --laps 5
    python scripts/record_dagger.py --policy residual --laps 5 --out data/dagger_residual.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from old_versions_drivers.project_V2 import RuleBasedDriver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient
from training.rl.features import FEATURE_DIM, build_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_EVERY = 50


def _make_policy(name: str):
    if name == "bc":
        from _DRIVER.driver import BCDriver
        return BCDriver()
    if name == "residual":
        from drivers.rl.residual_driver import ResidualRLDriver
        return ResidualRLDriver()
    raise ValueError(f"Unknown policy: {name}")


def run(
    policy_name: str,
    laps: int,
    host: Optional[str],
    port: Optional[int],
    out_path: Path,
) -> dict:
    policy = _make_policy(policy_name)
    oracle = RuleBasedDriver()

    feature_cols = [f"feat_{i}" for i in range(FEATURE_DIM)]
    fieldnames = feature_cols + [
        "oracle_steer", "oracle_accel", "oracle_brake",
        "rollout_steer", "rollout_accel", "rollout_brake",
        "lap", "distFromStart",
    ]

    rows: list[dict] = []
    lap_times: list[float] = []
    lap_count = 0
    # Giro (state.lap) all'ultima registrazione: rileva un nuovo giro anche se
    # due giri consecutivi hanno tempo identico (simulazione deterministica) —
    # stesso doppio criterio di scripts/evaluate_common.py.
    lap_at_last_record = 0
    total_steps = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush() -> None:
        # Scrive quanto raccolto finora, anche in caso di crash/timeout SCR a
        # metà corsa (osservato: la vettura BC può bloccarsi contro un muro e
        # far scadere il timeout SCR di TORCS, terminando la connessione).
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("DAgger dataset (flush): %s (%d rows)", out_path, len(rows))

    try:
        with TORCSClient(host=host, port=port) as client:
            logger.info(
                "Connected to TORCS. Rollout policy='%s', oracle=RuleBasedDriver, %d lap(s).",
                policy_name, laps,
            )

            while True:
                result = client.receive()

                if result == SHUTDOWN:
                    logger.info("Server shutdown.")
                    break
                if result == RESTART:
                    logger.info("Server restart signal.")
                    policy.on_restart()
                    oracle.on_restart()
                    lap_count = 0
                    lap_times = []
                    lap_at_last_record = 0
                    continue

                state = result
                rollout_action = policy.step(state)
                oracle_action = oracle.step(state)  # solo etichetta, non guida
                client.send(rollout_action)

                total_steps += 1

                feat = build_feature_vector(state)
                row = {f"feat_{i}": float(feat[i]) for i in range(FEATURE_DIM)}
                row.update({
                    "oracle_steer": oracle_action.steer,
                    "oracle_accel": oracle_action.accel,
                    "oracle_brake": oracle_action.brake,
                    "rollout_steer": rollout_action.steer,
                    "rollout_accel": rollout_action.accel,
                    "rollout_brake": rollout_action.brake,
                    "lap": state.lap,
                    "distFromStart": state.distFromStart,
                })
                rows.append(row)

                if total_steps % STATUS_EVERY == 0:
                    logger.info(
                        "lap %d | %6.0f m | %5.1f km/h | rollout steer %+.2f acc %.1f brk %.1f | "
                        "oracle steer %+.2f acc %.1f brk %.1f",
                        state.lap, state.distFromStart, state.speed,
                        rollout_action.steer, rollout_action.accel, rollout_action.brake,
                        oracle_action.steer, oracle_action.accel, oracle_action.brake,
                    )

                if state.lastLapTime > 0 and (
                    not lap_times
                    or state.lastLapTime != lap_times[-1]
                    or state.lap > lap_at_last_record
                ):
                    lap_times.append(state.lastLapTime)
                    lap_count += 1
                    lap_at_last_record = state.lap
                    logger.info("Lap %d completed in %.3f s", lap_count, state.lastLapTime)
                    if lap_count >= laps:
                        logger.info("Target laps reached — releasing control to TORCS.")
                        break
    finally:
        _flush()

    return {
        "policy": policy_name,
        "laps_completed": lap_count,
        "lap_times": lap_times,
        "total_steps": total_steps,
        "csv": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DAgger state-action pairs using RuleBasedDriver as oracle")
    parser.add_argument("--policy", choices=["bc", "residual"], default="bc")
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/dagger_<policy>_<ts>.csv)")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "data" / f"dagger_{args.policy}_{timestamp}.csv"

    run(policy_name=args.policy, laps=args.laps, host=args.host, port=args.port, out_path=out_path)


if __name__ == "__main__":
    main()
