"""Loop condiviso da scripts/run/run_agent.py e scripts/run/run_agent_rl.py.

receive/step/send, logging di stato periodico, telemetria opzionale su CSV,
e salvataggio del JSON di risultati — logica identica prima duplicata nei
due script, cambiava solo quale classe driver veniva istanziata.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATUS_EVERY = 50  # una riga di stato per ogni secondo simulato (~50 step/s)


def run_driver(
    driver,
    driver_name: str,
    laps: int = 1,
    host: Optional[str] = None,
    port: Optional[int] = None,
    save_telemetry: bool = False,
) -> dict:
    rows: list[dict] = []
    lap_times: list[float] = []
    lap_count = 0
    # Giro (state.lap) al momento dell'ultima registrazione in lap_times: usato
    # come guard anti-doppio-conteggio e per rilevare un nuovo giro anche
    # quando lastLapTime è identico al giro precedente (simulazione
    # deterministica a parità di codice: due giri consecutivi possono avere
    # lo stesso tempo al millesimo, e il solo confronto sul tempo non lo vedrebbe).
    lap_at_last_record = 0
    max_speed = 0.0
    max_damage = 0.0
    off_track_steps = 0
    total_steps = 0

    with TORCSClient(host=host, port=port) as client:
        logger.info("Connected to TORCS. Starting '%s' driver for %d lap(s).", driver_name, laps)

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
            max_damage = max(max_damage, state.damage)
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

            if save_telemetry:
                rows.append({
                    "timestamp": time.time(),
                    "distFromStart": state.distFromStart,
                    "distRaced": state.distRaced,
                    "curLapTime": state.curLapTime,
                    "angle": state.angle,
                    "speed": state.speed,
                    "speedY": state.speedY,
                    "speedZ": state.speedZ,
                    "trackPos": state.trackPos,
                    **{f"track_{i}": state.track[i] for i in range(len(state.track))},
                    "rpm": state.rpm,
                    "gear": state.gear,
                    "damage": state.damage,
                    **{f"wheel_{i}": state.wheelSpinVel[i] for i in range(len(state.wheelSpinVel))},
                    "steer": action.steer,
                    "accel": action.accel,
                    "brake": action.brake,
                })

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
                    # NON forzare uno shutdown meta=2 — interromperebbe la
                    # sessione prima che TORCS possa mostrare la schermata dei
                    # risultati del giro. Per una gara a giri limitati TORCS
                    # termina naturalmente una volta tagliato il traguardo;
                    # basta smettere di guidare e lasciare che mostri i risultati.
                    logger.info("Target laps reached — releasing control to TORCS.")
                    break

    off_track_pct = (off_track_steps / max(total_steps, 1)) * 100.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    telemetry_path: Optional[Path] = None
    if save_telemetry and rows:
        telemetry_path = PROJECT_ROOT / "data" / f"{driver_name}_{timestamp}.csv"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with telemetry_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Telemetry: %s (%d rows)", telemetry_path, len(rows))

    results = {
        "driver": driver_name,
        "laps_completed": lap_count,
        "lap_times": lap_times,
        "best_lap": min(lap_times) if lap_times else None,
        "max_speed_kmh": round(max_speed, 2),
        "off_track_pct": round(off_track_pct, 2),
        "damage": round(max_damage, 2),
        "total_steps": total_steps,
        "telemetry_csv": str(telemetry_path) if telemetry_path else None,
    }

    results_path = PROJECT_ROOT / "results" / f"{driver_name}_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results: %s", results_path)
    logger.info("Summary: best_lap=%.3fs  max=%.1f km/h  off_track=%.1f%%",
                results["best_lap"] or 0, max_speed, off_track_pct)
    return results
