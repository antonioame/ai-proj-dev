"""Collect RL experience by running episodes via launcher.

Usage:
    python training/rl/collect_experience.py --episodes 10 --output data/experience.pkl
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_episode(driver: str = "rule_based") -> dict:
    """Run one episode via launcher and return telemetry."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "launch_race.py"),
        "--driver", driver,
        "--laps", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"Episode failed: {result.stderr}")
        return None

    # Parse telemetry from CSV
    # (would need to implement CSV parsing here)
    return {"success": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RL experience via launcher")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", default="data/experience.pkl")
    args = parser.parse_args()

    experience = []
    for i in range(args.episodes):
        print(f"Episode {i + 1}/{args.episodes}...")
        ep = run_episode()
        if ep:
            experience.append(ep)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(experience, f)

    print(f"\nCollected {len(experience)} episodes → {output_path}")


if __name__ == "__main__":
    main()
