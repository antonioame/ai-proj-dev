"""PyTorch Dataset for behavioral cloning from TORCS telemetry CSV."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class TORCSDataset(Dataset):
    """Load TORCS telemetry CSV and expose (sensor_state, action) pairs."""

    def __init__(self, csv_path: str | Path):
        """Load CSV and normalize data.

        Parameters
        ----------
        csv_path : str | Path
            Path to telemetry CSV (e.g., data/keyboard_*.csv)
        """
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)

        # Sensor columns: angle, speed, speedY, speedZ, trackPos, track_0..18, rpm, gear
        sensor_cols = [
            'angle', 'speed', 'speedY', 'speedZ', 'trackPos',
            'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
            'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
            'track_10', 'track_11', 'track_12', 'track_13', 'track_14',
            'track_15', 'track_16', 'track_17', 'track_18',
            'rpm', 'gear'
        ]

        # Action columns: steer, accel, brake, gear_cmd
        action_cols = ['steer', 'accel', 'brake', 'gear_cmd']

        self.X = df[sensor_cols].values.astype(np.float32)
        self.Y = df[action_cols].values.astype(np.float32)

        # Normalize inputs (z-score)
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0) + 1e-6  # avoid division by zero
        self.X = (self.X - self.X_mean) / self.X_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])
