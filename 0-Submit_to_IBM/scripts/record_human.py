"""Record sensor + action telemetry to CSV for one lap.

Because SCR has no true observer mode, this script drives in "shadow mode":
it forwards a fixed neutral action while recording every sensor frame.
The intended workflow:

  Windows: launch TORCS headlessly with corkscrew_solo.xml
  Mac:     python scripts/record_human.py

The CSV is saved as data/human_YYYYMMDD_HHMMSS.csv.
One row = one simulation step (~50 ms).

CSV columns: timestamp, angle, speed, speedY, speedZ, trackPos,
             track_0 … track_18, rpm, gear, distRaced, curLapTime,
             steer, accel, brake, gear_cmd
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torcs_env.client import RESTART, SHUTDOWN, TORCSClient
from torcs_env.actions import Action

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _track_cols() -> list[str]:
    return [f"track_{i}" for i in range(19)]


FIELDNAMES = [
    "timestamp", "angle", "speed", "speedY", "speedZ", "trackPos",
    *_track_cols(),
    "rpm", "gear", "distRaced", "curLapTime",
    "steer", "accel", "brake", "gear_cmd",
]


class NeutralDriver:
    def step(self, _):
        return Action(accel=0.3, gear=1)

    def on_restart(self) -> None:
        pass


def record(host: str | None = None, port: int | None = None) -> Path:
    driver_name = "neutral"
    driver = NeutralDriver()
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"human_{timestamp_str}.csv"

    rows: list[dict] = []
    lap_start: float | None = None
    lap_completed = False

    with TORCSClient(host=host, port=port) as client:
        logger.info("Connected. Recording one lap with '%s' driver.", driver_name)

        while not lap_completed:
            result = client.receive()

            if result == SHUTDOWN:
                logger.info("Server shutdown.")
                break
            if result == RESTART:
                driver.on_restart()
                rows.clear()
                lap_start = None
                continue

            state = result
            action = driver.step(state)
            client.send(action)

            now = time.time()
            if lap_start is None:
                lap_start = now

            row = {
                "timestamp": now,
                "angle": state.angle,
                "speed": state.speed,
                "speedY": state.speedY,
                "speedZ": state.speedZ,
                "trackPos": state.trackPos,
                **{f"track_{i}": state.track[i] for i in range(min(19, len(state.track)))},
                "rpm": state.rpm,
                "gear": state.gear,
                "distRaced": state.distRaced,
                "curLapTime": state.curLapTime,
                "steer": action.steer,
                "accel": action.accel,
                "brake": action.brake,
                "gear_cmd": action.gear,
            }
            rows.append(row)

            # Detect lap completion
            if state.lastLapTime > 0 and rows:
                lap_time = state.lastLapTime
                logger.info("Lap completed in %.3f s", lap_time)
                lap_completed = True

    # Write CSV
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows to %s", len(rows), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one lap of TORCS telemetry (neutral shadow driver)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    record(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
