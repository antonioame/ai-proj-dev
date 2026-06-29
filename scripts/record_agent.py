"""Record a lap with any registered driver and save telemetry to CSV.

Usage:
    python scripts/record_agent.py --driver rule_based [--laps 1] [--host HOST] [--port PORT]

Output: data/recorded_<driver>_<timestamp>.csv
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

from drivers.registry import load_driver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIELDNAMES = [
    "timestamp", "angle", "speed", "speedY", "speedZ", "trackPos",
    *[f"track_{i}" for i in range(19)],
    "rpm", "gear", "distFromStart", "distRaced", "curLapTime",
    "steer", "accel", "brake", "gear_cmd",
]


def record(
    driver_name: str = "rule_based",
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
) -> Path:
    driver = load_driver(driver_name)

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"recorded_{driver_name}_{ts}.csv"

    rows: list[dict] = []
    laps_done = 0
    lap_complete_step = -1
    step = 0

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
                    driver.reset()
                    continue

                state = result
                action = driver.step(state)
                client.send(action)
                step += 1

                rows.append({
                    "timestamp": time.time(),
                    "angle": state.angle,
                    "speed": state.speed,
                    "speedY": state.speedY,
                    "speedZ": state.speedZ,
                    "trackPos": state.trackPos,
                    **{f"track_{i}": state.track[i] for i in range(min(19, len(state.track)))},
                    "rpm": state.rpm,
                    "gear": state.gear,
                    "distFromStart": state.distFromStart,
                    "distRaced": state.distRaced,
                    "curLapTime": state.curLapTime,
                    "steer": action.steer,
                    "accel": action.accel,
                    "brake": action.brake,
                    "gear_cmd": action.gear,
                })

                if len(rows) % 50 == 0:
                    logger.info(
                        "t=%.1f  %.1f km/h  gear %d  pos %.2f  steer %.2f  acc %.1f  brk %.1f",
                        state.curLapTime, state.speed, state.gear, state.trackPos,
                        action.steer, action.accel, action.brake,
                    )

                if state.lastLapTime > 0 and lap_complete_step < 0:
                    laps_done += 1
                    logger.info("Lap %d done in %.1f s", laps_done, state.lastLapTime)
                    lap_complete_step = step

                # Allow ~1 s of data after lap ends before stopping / resetting
                if lap_complete_step >= 0 and (step - lap_complete_step) >= 50:
                    if laps_done >= laps:
                        break
                    lap_complete_step = -1

    except Exception as exc:
        logger.warning("Connection lost: %s", exc)
    finally:
        driver.reset()

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows → %s", len(rows), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record agent telemetry for a lap")
    parser.add_argument("--driver", default="rule_based",
                        help="Driver: rule_based | optimal")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    record(driver_name=args.driver, laps=args.laps, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
