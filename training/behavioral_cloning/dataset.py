"""PyTorch Dataset for behavioral-cloning training from CSV telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Sensor columns used as model inputs
# Supports both old format (speedX, damage) and new format (speed, no damage)
SENSOR_COLS = ["speed", "trackPos", "angle", "rpm", "gear"]

# Action columns used as model outputs
ACTION_COLS = ["steer", "accel", "brake", "gear_cmd"]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a recorded telemetry CSV."""
    return pd.read_csv(path)


class TelemetryDataset(Dataset):
    """Map-style Dataset over one or more telemetry CSV files."""

    def __init__(
        self,
        csv_paths: list[str | Path],
        sensor_cols: Optional[list[str]] = None,
        action_cols: Optional[list[str]] = None,
        normalise: bool = True,
    ) -> None:
        sensor_cols = sensor_cols or SENSOR_COLS
        action_cols = action_cols or ACTION_COLS

        frames = [load_csv(p) for p in csv_paths]
        df = pd.concat(frames, ignore_index=True).dropna()

        # Rename output gear column to avoid collision with input gear
        if "gear" in df.columns and "gear_out" not in df.columns:
            df["gear_out"] = df["gear"]

        self._sensors = df[sensor_cols].values.astype(np.float32)
        self._actions = df[action_cols].values.astype(np.float32)

        if normalise:
            self._sensor_mean = self._sensors.mean(axis=0)
            self._sensor_std = self._sensors.std(axis=0) + 1e-8
            self._sensors = (self._sensors - self._sensor_mean) / self._sensor_std
        else:
            self._sensor_mean = np.zeros(len(sensor_cols), dtype=np.float32)
            self._sensor_std = np.ones(len(sensor_cols), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._sensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._sensors[idx])
        y = torch.from_numpy(self._actions[idx])
        return x, y

    @property
    def input_dim(self) -> int:
        return self._sensors.shape[1]

    @property
    def output_dim(self) -> int:
        return self._actions.shape[1]

    @property
    def normalisation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) arrays for inference normalisation."""
        return self._sensor_mean, self._sensor_std
