"""Esegue il driver candidato bc_tita_v1 (data_collection/tita/driver_candidate.py)
contro un server TORCS gia' avviato, con log di stato per riga come
scripts/run/run_agent.py. Utile per osservare la guida dal vivo (GUI) prima/oltre
alla valutazione JSON di evaluate_tita_candidate.py.

Non tocca _DRIVER/, drivers/, scripts/.

Usage:
    python data_collection/tita/run_tita_candidate.py --laps 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_collection.tita.driver_candidate import BCTitaCandidateDriver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATUS_EVERY = 50  # una riga di stato per ogni secondo simulato (50 step/s circa)


def run(
    laps: int = 1,
    host: Optional[str] = None,
    port: Optional[int] = None,
    checkpoint_name: str = "bc_tita_v1",
) -> dict:
    candidate_dir = Path(__file__).resolve().parent / "candidate_models"
    driver = BCTitaCandidateDriver(
        checkpoint_path=candidate_dir / f"{checkpoint_name}.pth",
        stats_path=candidate_dir / f"{checkpoint_name}.npz",
    )

    lap_times: list[float] = []
    lap_count = 0
    # Giro (state.lap) all'ultima registrazione: rileva un nuovo giro anche se
    # due giri consecutivi hanno tempo identico (simulazione deterministica):
    # stesso doppio criterio di scripts/eval/evaluate_common.py.
    lap_at_last_record = 0
    max_speed = 0.0
    off_track_steps = 0
    total_steps = 0

    with TORCSClient(host=host, port=port) as client:
        logger.info("Connected to TORCS. Starting '%s' driver for %d lap(s).", checkpoint_name, laps)

        while True:
            result = client.receive()

            if result == SHUTDOWN:
                logger.info("Server shutdown.")
                break
            if result == RESTART:
                logger.info("Server restart signal.")
                driver.on_restart()
                lap_count = 0
                lap_times = []
                lap_at_last_record = 0
                continue

            state = result
            action = driver.step(state)
            client.send(action)

            total_steps += 1
            max_speed = max(max_speed, state.speed)
            on_track = abs(state.trackPos) <= 1.0
            if not on_track:
                off_track_steps += 1

            if total_steps % STATUS_EVERY == 0:
                logger.info(
                    "lap %d | %6.0f m | %5.1f km/h | gear %d | rpm %4.0f | "
                    "pos %+.2f%s | steer %+.2f acc %.1f brk %.1f",
                    state.lap, state.distFromStart, state.speed, state.gear,
                    state.rpm, state.trackPos, "" if on_track else " OFF",
                    action.steer, action.accel, action.brake,
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
                    logger.info("Target laps reached, releasing control to TORCS.")
                    break

    off_track_pct = (off_track_steps / max(total_steps, 1)) * 100.0
    results = {
        "driver": checkpoint_name,
        "laps_completed": lap_count,
        "lap_times": lap_times,
        "best_lap": min(lap_times) if lap_times else None,
        "max_speed_kmh": round(max_speed, 2),
        "off_track_pct": round(off_track_pct, 2),
        "total_steps": total_steps,
    }
    logger.info("Summary: best_lap=%.3fs  max=%.1f km/h  off_track=%.1f%%",
                results["best_lap"] or 0, max_speed, off_track_pct)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bc_tita_v1 candidate driver")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="bc_tita_v1")
    args = parser.parse_args()
    run(laps=args.laps, host=args.host, port=args.port, checkpoint_name=args.checkpoint_name)


if __name__ == "__main__":
    main()
