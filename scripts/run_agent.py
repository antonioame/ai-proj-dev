"""Run any registered driver against a TORCS server.

Usage:
    python scripts/run_agent.py --driver rule_based [--laps 1] [--host localhost] [--port 3001]
    python -m scripts.run_agent --driver rule_based
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drivers.base_driver import BaseDriver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# How often (in simulation steps) to log a live car-status line. The SCR server
# ticks at ~50 steps/s, so 50 ≈ one status line per simulated second.
STATUS_EVERY = 50


def load_driver(name: str) -> BaseDriver:
    if name == "rule_based":
        from drivers.rule_based.driver import RuleBasedDriver
        return RuleBasedDriver()
    elif name == "bc_model":
        from drivers.bc.driver import BCDriver
        return BCDriver()
    elif name == "optimal":
        from drivers.optimal.driver import OptimalLineDriver
        return OptimalLineDriver()
    elif name == "rl_model":
        from drivers.rl.driver import RLDriver
        return RLDriver()
    elif name.startswith("rl_"):
        # rl_ppo, rl_ddpg, rl_direct_v1, rl_persistent_v1, etc.
        # Detect algorithm from model name or suffix
        suffix = name.split("_", 1)[1]

        # Check if it's a known pattern (ppo, ddpg) or a custom model name
        if suffix in ("ppo", "ddpg"):
            model_path = f"models/{suffix}_v1"
            algo = suffix
        else:
            # Assume custom model name, default to PPO
            model_path = f"models/{suffix}/final" if (PROJECT_ROOT / "models" / suffix / "final.zip").exists() else f"models/{name}"
            algo = "ppo"

        from drivers.rl.driver import RLDriver
        return RLDriver(model_path=model_path, algo=algo)
    raise ValueError(
        f"Unknown driver '{name}'. Available: rule_based, bc_model, optimal, rl_model, rl_ppo, rl_ddpg, or rl_<model_name>"
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
            on_track = abs(state.trackPos) <= 1.0
            if not on_track:
                off_track_steps += 1

            # Live car/circuit status so the run can be observed in real time.
            if total_steps % STATUS_EVERY == 0:
                logger.info(
                    "lap %d | %6.0f m | %6.1f km/h | gear %d | rpm %4.0f | "
                    "trackPos %+5.2f%s | angle %+5.2f | steer %+4.2f acc %3.1f brk %3.1f",
                    state.lap, state.distFromStart, state.speed, state.gear,
                    state.rpm, state.trackPos, "" if on_track else " OFF",
                    state.angle, action.steer, action.accel, action.brake,
                )

            if save_telemetry:
                rows.append({
                    "timestamp": time.time(),
                    "distFromStart": state.distFromStart,
                    "distRaced": state.distRaced,
                    "curLapTime": state.curLapTime,
                    "angle": state.angle,
                    "speed": state.speed,
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    telemetry_path: Optional[Path] = None
    if save_telemetry and rows:
        telemetry_path = PROJECT_ROOT / "data" / f"{driver_name}_{timestamp}.csv"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with telemetry_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Telemetry saved: %s (%d rows)", telemetry_path, len(rows))
    elif save_telemetry:
        logger.warning("Telemetry requested but no rows were recorded.")

    results = {
        "driver": driver_name,
        "laps_completed": lap_count,
        "lap_times": lap_times,
        "best_lap": min(lap_times) if lap_times else None,
        "max_speed_kmh": round(max_speed, 2),
        "off_track_pct": round(off_track_pct, 2),
        "total_steps": total_steps,
        "telemetry_csv": str(telemetry_path) if telemetry_path else None,
    }

    results_path = PROJECT_ROOT / "results" / f"{driver_name}_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved: %s", results_path)

    logger.info("Run complete: %s", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TORCS driver agent")
    parser.add_argument(
        "--driver", default="rule_based",
        help="Driver name: rule_based, bc_model, optimal, rl_model, rl_ddpg, rl_ppo"
    )
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
