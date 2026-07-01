"""Benchmark a driver over K laps and append metrics to laptime_ledger.csv.

Usage:
    python scripts/benchmark.py [--laps 3] [--config-id my_config] [--compare baseline_rule_based] [--notes "ABS added"]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = PROJECT_ROOT / "laptime_ledger.csv"
LEDGER_FIELDS = [
    "config_id", "git_sha",
    "best_lap_s", "median_lap_s", "top_speed_kmh",
    "off_track_pct", "damage", "valid", "notes",
]


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _run_race(laps: int) -> dict:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "launch_race.py"),
        "--laps", str(laps),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    # find the freshest JSON result for the BC driver
    results_dir = PROJECT_ROOT / "results"
    candidates = sorted(results_dir.glob("bc_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No results JSON found in {results_dir} after race")
    return json.loads(candidates[-1].read_text())


def _append_ledger(row: dict) -> None:
    write_header = not LEDGER_PATH.exists()
    with LEDGER_PATH.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    with LEDGER_PATH.open() as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description="K-lap benchmark of the BC driver → laptime_ledger.csv")
    parser.add_argument("--laps", type=int, default=1, help="Laps per session")
    parser.add_argument("--sessions", type=int, default=1,
                        help="Independent race sessions (each is a standing-start lap)")
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--compare", default=None, help="config_id to compare against in ledger")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config_id = args.config_id or f"bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    all_lap_times: list[float] = []
    max_speed = 0.0
    off_track_pct = 0.0
    damage_total = 0.0

    for s in range(args.sessions):
        print(f"\n--- Session {s+1}/{args.sessions} ---")
        r = _run_race(args.laps)
        lap_times: list[float] = r.get("lap_times", [])
        all_lap_times.extend(lap_times)
        max_speed = max(max_speed, r.get("max_speed_kmh", 0.0))
        off_track_pct = max(off_track_pct, r.get("off_track_pct", 0.0))

    valid_times = [t for t in all_lap_times if t > 0]
    best = min(valid_times) if valid_times else None
    med = statistics.median(valid_times) if valid_times else None
    valid = best is not None and off_track_pct < 5.0 and damage_total == 0

    row = {
        "config_id": config_id,
        "git_sha": _git_sha(),
        "best_lap_s": f"{best:.3f}" if best else "",
        "median_lap_s": f"{med:.3f}" if med else "",
        "top_speed_kmh": f"{max_speed:.2f}",
        "off_track_pct": f"{off_track_pct:.2f}",
        "damage": f"{damage_total:.0f}",
        "valid": "true" if valid else "false",
        "notes": args.notes,
    }
    _append_ledger(row)

    print(f"\n=== Benchmark results: '{config_id}' ===")
    print(f"  Best lap:   {best:.3f} s" if best else "  Best lap:   N/A")
    print(f"  Median lap: {med:.3f} s" if med else "  Median lap: N/A")
    print(f"  Top speed:  {max_speed:.1f} km/h")
    print(f"  Off-track:  {off_track_pct:.2f}%")
    print(f"  Valid:      {valid}")

    if args.compare:
        records = _load_ledger()
        ref_rows = [r for r in records if r.get("config_id") == args.compare]
        if ref_rows:
            ref = ref_rows[-1]
            ref_best = float(ref["best_lap_s"]) if ref.get("best_lap_s") else None
            if ref_best and best:
                delta = best - ref_best
                sign = "+" if delta >= 0 else ""
                pct = delta / ref_best * 100
                print(f"\n  vs '{args.compare}': {sign}{delta:.3f} s  ({sign}{pct:.2f}%)")
        else:
            print(f"  (no ledger entry for '{args.compare}')")

    print(f"\nLedger: {LEDGER_PATH}")


if __name__ == "__main__":
    main()
