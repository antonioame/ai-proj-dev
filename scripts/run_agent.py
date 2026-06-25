"""Run any registered driver against a TORCS server.

Usage:
    python scripts/run_agent.py --driver rule_based [--laps 1] [--host localhost] [--port 3001]
    python -m scripts.run_agent --driver rule_based
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Make project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drivers.base_driver import BaseDriver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_driver(name: str) -> BaseDriver:
    if name == "rule_based":
        from drivers.rule_based.driver import RuleBasedDriver
        return RuleBasedDriver()
    raise ValueError(
        f"Unknown driver '{name}'. Available: rule_based"
    )


def run(
    driver_name: str,
    laps: int = 1,
    host: Optional[str] = None,
    port: Optional[int] = None,
    save_telemetry: bool = False,
) -> dict:
    driver = load_driver(driver_name)

    rows: list[dict] = []
    lap_times: list[float] = []
    lap_count = 0
    max_speed = 0.0
    off_track_steps = 0
    total_steps = 0

    with TORCSClient(host=host, port=port) as client:
        logger.info("Connected to TORCS. Starting %s driver for %d lap(s).", driver_name, laps)

        while True:
            result = client.receive()

            if result == SHUTDOWN:
                logger.info("Server shutdown.")
                break
            if result == RESTART:
                logger.info("Server restart signal received.")
                driver.on_restart()
                lap_count = 0
                continue

            state = result
            action = driver.step(state)

            total_steps += 1
            max_speed = max(max_speed, state.speed)
            if abs(state.trackPos) > 1.0:
                off_track_steps += 1

            if save_telemetry:
                rows.append({
                    "timestamp": time.time(),
                    "angle": state.angle,
                    "speed": state.speed,
                    "trackPos": state.trackPos,
                    **{f"track_{i}": state.track[i] for i in range(len(state.track))},
                    "rpm": state.rpm,
                    "gear": state.gear,
                    "steer": action.steer,
                    "accel": action.accel,
                    "brake": action.brake,
                })

            # Detect lap completion; send shutdown *instead* of the normal action
            # so TORCS receives a single clean exit signal for the final step.
            if state.lastLapTime > 0 and (not lap_times or state.lastLapTime != lap_times[-1]):
                lap_times.append(state.lastLapTime)
                lap_count += 1
                logger.info("Lap %d completed in %.3f s", lap_count, state.lastLapTime)
                if lap_count >= laps:
                    client.send_shutdown()
                    break

            client.send(action)

    off_track_pct = (off_track_steps / max(total_steps, 1)) * 100.0
    avg_speed = sum(lap_times) / len(lap_times) if lap_times else 0.0

    results = {
        "driver": driver_name,
        "laps_completed": lap_count,
        "lap_times": lap_times,
        "best_lap": min(lap_times) if lap_times else None,
        "max_speed_kmh": round(max_speed, 2),
        "off_track_pct": round(off_track_pct, 2),
        "total_steps": total_steps,
    }

    logger.info("Run complete: %s", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TORCS driver agent")
    parser.add_argument("--driver", default="rule_based", help="Driver name")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--telemetry", action="store_true", help="Save telemetry to CSV")
    args = parser.parse_args()

    run(
        driver_name=args.driver,
        laps=args.laps,
        host=args.host,
        port=args.port,
        save_telemetry=args.telemetry,
    )


if __name__ == "__main__":
    main()
