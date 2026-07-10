"""Valuta il driver BC (_DRIVER/) e salva un JSON di risultati strutturato.

Usage:
    python scripts/evaluate.py [--laps 1] [--output results/eval.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _DRIVER.driver import BCDriver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: Path | None = None,
) -> dict:
    driver_name = "bc"
    driver = BCDriver()

    lap_times: list[float] = []
    speed_samples: list[float] = []
    off_track_steps = 0
    total_steps = 0
    total_damage = 0.0
    lap_count = 0

    with TORCSClient(host=host, port=port) as client:
        logger.info("Evaluating '%s' for %d lap(s).", driver_name, laps)

        while True:
            result = client.receive()

            if result == SHUTDOWN:
                break
            if result == RESTART:
                driver.on_restart()
                continue

            state = result
            action = driver.step(state)
            client.send(action)

            total_steps += 1
            speed_samples.append(state.speed)
            total_damage = max(total_damage, state.damage)

            if abs(state.trackPos) > 1.0:
                off_track_steps += 1

            # Rileva un nuovo tempo sul giro
            if state.lastLapTime > 0 and (
                not lap_times or state.lastLapTime != lap_times[-1]
            ):
                lap_times.append(state.lastLapTime)
                lap_count += 1
                logger.info("Lap %d: %.3f s", lap_count, state.lastLapTime)
                if lap_count >= laps:
                    break

    off_track_pct = (off_track_steps / max(total_steps, 1)) * 100.0
    avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 0.0
    max_speed = max(speed_samples) if speed_samples else 0.0

    results = {
        "driver": driver_name,
        "evaluated_at": datetime.now().isoformat(),
        "laps_requested": laps,
        "laps_completed": lap_count,
        "lap_times_s": lap_times,
        "best_lap_s": min(lap_times) if lap_times else None,
        "avg_lap_s": sum(lap_times) / len(lap_times) if lap_times else None,
        "max_speed_kmh": round(max_speed, 2),
        "avg_speed_kmh": round(avg_speed, 2),
        "off_track_pct": round(off_track_pct, 2),
        "damage": round(total_damage, 2),
        "total_steps": total_steps,
    }

    logger.info("Results: %s", json.dumps(results, indent=2))

    if output_path is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"eval_{driver_name}_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    return results


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
