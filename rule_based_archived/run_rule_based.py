"""Minimal standalone runner for the archived rule_based driver.

Not wired into the main bc_driver pipeline (scripts/run_agent.py etc).
Usage:
    python rule_based_archived/run_rule_based.py [--laps 1] [--host HOST] [--port PORT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rule_based_archived.driver import RuleBasedDriver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the archived rule_based driver")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    driver = RuleBasedDriver()
    lap_times: list[float] = []

    with TORCSClient(host=args.host, port=args.port) as client:
        print(f"Connected. Running rule_based for {args.laps} lap(s).")
        while True:
            result = client.receive()

            if result == SHUTDOWN:
                print("Server shutdown.")
                break
            if result == RESTART:
                driver.on_restart()
                lap_times.clear()
                continue

            state = result
            action = driver.step(state)
            client.send(action)

            if state.lastLapTime > 0 and (not lap_times or state.lastLapTime != lap_times[-1]):
                lap_times.append(state.lastLapTime)
                print(f"Lap {len(lap_times)} completed in {state.lastLapTime:.3f} s")
                if len(lap_times) >= args.laps:
                    print("Target laps reached — releasing control to TORCS.")
                    break


if __name__ == "__main__":
    main()
