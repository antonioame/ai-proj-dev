"""Record lap telemetry from any agent driver."""

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


def record_agent(driver_name: str = "rule_based", laps: int = 1, host: str | None = None, port: int | None = None) -> Path:
    """Record telemetry from an agent driver.

    Parameters
    ----------
    driver_name : str
        Name of driver to use: 'rule_based', 'optimal', 'bc_model'
    laps : int
        Number of laps to record
    host : str | None
        TORCS server host (default: localhost)
    port : int | None
        TORCS server port (default: 3001)
    """
    # Dynamically import the driver
    if driver_name == "rule_based":
        from drivers.rule_based.driver import RuleBasedDriver
        driver = RuleBasedDriver()
    elif driver_name == "optimal":
        from drivers.optimal.driver import OptimalLineDriver
        driver = OptimalLineDriver()
    elif driver_name == "bc_model":
        from drivers.bc.driver import BCDriver
        driver = BCDriver()
    else:
        raise ValueError(f"Unknown driver: {driver_name}")

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"recorded_{driver_name}_{timestamp_str}.csv"

    rows: list[dict] = []
    laps_completed = 0
    lap_start: float | None = None
    lap_complete_step = -1
    step = 0

    try:
        with TORCSClient(host=host, port=port) as client:
            logger.info(f"Connected to TORCS. Recording {laps} lap(s) with {driver_name} driver.")

            while laps_completed < laps:
                result = client.receive()

                if result == SHUTDOWN:
                    logger.info("Server shutdown.")
                    break
                if result == RESTART:
                    logger.info("Race restarted.")
                    rows.clear()
                    laps_completed = 0
                    lap_start = None
                    lap_complete_step = -1
                    step = 0
                    driver.reset()
                    continue

                state = result
                action = driver.step(state)
                client.send(action)
                step += 1

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

                # Live status every ~1 second (20 steps @ 50ms each)
                if len(rows) % 20 == 0:
                    logger.info(
                        "time=%.1f speed=%.1f gear=%d trackPos=%.2f angle=%.2f "
                        "steer=%.2f accel=%.2f brake=%.2f",
                        state.curLapTime, state.speed, state.gear, state.trackPos,
                        state.angle, action.steer, action.accel, action.brake
                    )

                # Detect lap completion
                if state.lastLapTime > 0 and rows and lap_complete_step < 0:
                    lap_time = state.lastLapTime
                    laps_completed += 1
                    logger.info("✓ Lap %d completed in %.1f s", laps_completed, lap_time)
                    lap_complete_step = step

                # Wait 20 more steps (~1 second) after lap completion before continuing
                if lap_complete_step >= 0 and (step - lap_complete_step) >= 20:
                    if laps_completed >= laps:
                        break
                    else:
                        lap_start = None
                        lap_complete_step = -1

    except Exception as e:
        logger.warning("Connection lost: %s", e)
    finally:
        driver.reset()

    # Write CSV
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows to %s", len(rows), out_path)
    return out_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Record agent-driven laps with telemetry")
    parser.add_argument("--driver", default="rule_based", help="Driver to use: rule_based, optimal, bc_model")
    parser.add_argument("--laps", type=int, default=1, help="Number of laps to record")
    parser.add_argument("--host", default=None, help="TORCS server host (default: localhost)")
    parser.add_argument("--port", type=int, default=None, help="TORCS server port (default: 3001)")
    args = parser.parse_args()
    record_agent(driver_name=args.driver, laps=args.laps, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
