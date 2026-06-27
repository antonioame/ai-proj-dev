"""Build track_map.json from a telemetry CSV (recorded with --telemetry flag).

Usage:
    python scripts/build_track_map.py --telemetry data/rule_based_YYYYMMDD.csv [--output torcs_env/track_data/track_map.json]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torcs_env.track_map import (
    TRACK_LENGTH_M, BUCKET_M, CORNER_CURVATURE_THRESH, Bucket, TrackMap
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "torcs_env" / "track_data" / "track_map.json"


def _curvature(row: dict) -> float:
    """Signed curvature from track sensor asymmetry.

    Sensors 2-4 look left, sensors 14-16 look right.
    Positive = left sees further = right-hand corner.
    """
    try:
        left  = (float(row["track_2"]) + float(row["track_3"]) + float(row["track_4"])) / 3.0
        right = (float(row["track_14"]) + float(row["track_15"]) + float(row["track_16"])) / 3.0
        total = left + right
        return (left - right) / total if total > 1.0 else 0.0
    except (KeyError, ValueError, ZeroDivisionError):
        return 0.0


def _smooth(values: list[float], window: int = 6) -> list[float]:
    """Circular moving average."""
    n = len(values)
    result: list[float] = []
    for i in range(n):
        s = sum(values[(i + j) % n] for j in range(-window, window + 1))
        result.append(s / (2 * window + 1))
    return result


def build_track_map(telemetry_path: Path) -> TrackMap:
    buckets_speed:   dict[int, list[float]] = defaultdict(list)
    buckets_trackPos: dict[int, list[float]] = defaultdict(list)
    buckets_curv:    dict[int, list[float]] = defaultdict(list)

    with telemetry_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                dist      = float(row["distFromStart"])
                cur_lap   = float(row.get("curLapTime", 0.0))
                speed     = float(row["speed"])
                trackPos  = float(row["trackPos"])
            except (KeyError, ValueError):
                continue

            # skip pre-race (negative curLapTime) or beyond track length
            if cur_lap < 0 or dist > TRACK_LENGTH_M * 1.02:
                continue
            # skip very first steps where speed is effectively zero (launch stutter)
            if speed < 1.0:
                continue

            idx = int(dist / BUCKET_M)
            buckets_speed[idx].append(speed)
            buckets_trackPos[idx].append(trackPos)
            buckets_curv[idx].append(_curvature(row))

    n_buckets = int(math.ceil(TRACK_LENGTH_M / BUCKET_M))
    raw_curvatures: list[float] = []
    raw_speeds:     list[float] = []
    raw_trackPos:   list[float] = []

    prev_speed   = 80.0
    prev_trackPos = 0.0
    prev_curv    = 0.0

    for i in range(n_buckets):
        if buckets_speed.get(i):
            # Use MINIMUM speed as the floor (captures the slowest point in the bucket,
            # which is what the backward-pass trajectory needs as a constraint).
            min_speed    = min(buckets_speed[i])
            avg_trackPos = sum(buckets_trackPos[i]) / len(buckets_trackPos[i])
            avg_curv     = sum(buckets_curv[i]) / len(buckets_curv[i])
            prev_speed    = min_speed
            prev_trackPos = avg_trackPos
            prev_curv     = avg_curv
        else:
            min_speed     = prev_speed
            avg_trackPos  = prev_trackPos
            avg_curv      = prev_curv

        raw_speeds.append(min_speed)
        raw_trackPos.append(avg_trackPos)
        raw_curvatures.append(avg_curv)

    smooth_curv     = _smooth(raw_curvatures, window=6)
    smooth_speeds   = _smooth(raw_speeds,     window=2)
    smooth_trackPos = _smooth(raw_trackPos,   window=3)

    buckets: list[Bucket] = []
    for i in range(n_buckets):
        buckets.append(Bucket(
            s          = (i + 0.5) * BUCKET_M,
            speed_kmh  = smooth_speeds[i],
            trackPos   = smooth_trackPos[i],
            curvature  = smooth_curv[i],
            is_corner  = abs(smooth_curv[i]) > CORNER_CURVATURE_THRESH,
        ))

    return TrackMap(buckets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build track map JSON from telemetry CSV")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    tel = Path(args.telemetry)
    if not tel.exists():
        print(f"ERROR: telemetry file not found: {tel}")
        sys.exit(1)

    print(f"Building track map from {tel} ...")
    track_map = build_track_map(tel)

    out = Path(args.output)
    track_map.save(out)

    n_corners = sum(1 for b in track_map.buckets if b.is_corner)
    n_buckets = len(track_map.buckets)
    print(f"  Track length : {track_map.track_length:.1f} m")
    print(f"  Buckets      : {n_buckets}  ({BUCKET_M}m resolution)")
    print(f"  Corner buckets: {n_corners} / {n_buckets} ({100*n_corners/n_buckets:.0f}%)")
    print(f"  Min speed    : {min(b.speed_kmh for b in track_map.buckets):.1f} km/h")
    print(f"  Max curvature: {max(abs(b.curvature) for b in track_map.buckets):.3f}")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
