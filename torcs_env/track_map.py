"""Track map: per-distFromStart data bucketed from telemetry.

Build with:  python scripts/build_track_map.py --telemetry data/<file>.csv
Load with:   TrackMap.load(path)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

TRACK_LENGTH_M: float = 3608.4
BUCKET_M: float = 5.0
CORNER_CURVATURE_THRESH: float = 0.05  # |curvature| above this = corner


@dataclass
class Bucket:
    s: float           # distFromStart bucket centre (m)
    speed_kmh: float   # average speed seen in this bucket
    trackPos: float    # average track position in this bucket
    curvature: float   # signed sensor asymmetry (positive = right-hand corner)
    is_corner: bool    # |curvature| > CORNER_CURVATURE_THRESH


class TrackMap:
    def __init__(self, buckets: list[Bucket], track_length: float = TRACK_LENGTH_M) -> None:
        self.buckets = buckets
        self.track_length = track_length
        self._n = len(buckets)

    def _idx(self, s: float) -> int:
        s_mod = s % self.track_length
        return min(int(s_mod / BUCKET_M), self._n - 1)

    def lookup(self, s: float) -> Bucket:
        return self.buckets[self._idx(s)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "track_length_m": self.track_length,
            "bucket_m": BUCKET_M,
            "buckets": [asdict(b) for b in self.buckets],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TrackMap":
        data = json.loads(path.read_text())
        buckets = [Bucket(**b) for b in data["buckets"]]
        return cls(buckets, track_length=data["track_length_m"])
