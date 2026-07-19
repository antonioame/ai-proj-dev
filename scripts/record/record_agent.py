"""Registra un giro con il driver BC (_DRIVER/) e salva la telemetria in CSV.

Usage:
    python scripts/record/record_agent.py [--laps 1] [--host HOST] [--port PORT]

Output: data/recorded_bc_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _DRIVER.driver import BCDriver
from telemetry_row import TRACK_COLS, build_row
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DRIVER_NAME = "bc"

FIELDNAMES = [
    "timestamp", "angle", "speed", "speedY", "speedZ", "trackPos",
    *TRACK_COLS,
    "rpm", "gear", "distFromStart", "distRaced", "curLapTime",
    "steer", "accel", "brake", "gear_cmd",
]


def record(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
) -> Path:
    driver_name = DRIVER_NAME
    driver = BCDriver()

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"recorded_{driver_name}_{ts}.csv"

    rows: list[dict] = []
    laps_done = 0
    lap_complete_step = -1
    step = 0
    # Tempo/numero giro registrati all'ultimo incremento di laps_done: servono a
    # distinguere un nuovo giro davvero completato dal semplice permanere di
    # state.lastLapTime > 0 dopo che lap_complete_step è stato riazzerato.
    last_recorded_lap_time = 0.0
    last_recorded_lap = 0

    try:
        with TORCSClient(host=host, port=port) as client:
            logger.info("Connected. Recording %d lap(s) with '%s'.", laps, driver_name)

            while laps_done < laps:
                result = client.receive()

                if result == SHUTDOWN:
                    logger.info("Server shutdown.")
                    break
                if result == RESTART:
                    logger.info("Race restarted — clearing recorded data.")
                    rows.clear()
                    laps_done = 0
                    lap_complete_step = -1
                    step = 0
                    last_recorded_lap_time = 0.0
                    last_recorded_lap = 0
                    driver.on_restart()
                    continue

                state = result
                action = driver.step(state)
                client.send(action)
                step += 1

                rows.append(build_row(time.time(), state, action, include_dist_from_start=True))

                if len(rows) % 50 == 0:
                    logger.info(
                        "t=%.1f  %.1f km/h  gear %d  pos %.2f  steer %.2f  acc %.1f  brk %.1f",
                        state.curLapTime, state.speed, state.gear, state.trackPos,
                        action.steer, action.accel, action.brake,
                    )

                # Doppio criterio per un giro davvero nuovo: lastLapTime non aggiorna
                # più dopo il primo giro (resta > 0 per sempre), quindi confrontarlo
                # da solo non basta a rilevare il secondo giro se il tempo è identico
                # (possibile in simulazione deterministica) — state.lap (derivato dai
                # reset di distRaced nel client) copre quel caso.
                if state.lastLapTime > 0 and lap_complete_step < 0 and (
                    state.lastLapTime != last_recorded_lap_time or state.lap != last_recorded_lap
                ):
                    laps_done += 1
                    last_recorded_lap_time = state.lastLapTime
                    last_recorded_lap = state.lap
                    logger.info("Lap %d done in %.1f s", laps_done, state.lastLapTime)
                    lap_complete_step = step

                # Consente ~1 s di dati dopo la fine del giro prima di fermarsi/resettare
                if lap_complete_step >= 0 and (step - lap_complete_step) >= 50:
                    if laps_done >= laps:
                        break
                    lap_complete_step = -1

    except Exception as exc:
        logger.warning("Connection lost: %s", exc)
    finally:
        driver.on_restart()

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows → %s", len(rows), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record BC driver telemetry for a lap")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    record(laps=args.laps, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
