"""Behavioral Cloning driver — runs trained BC policy on TORCS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from drivers.base_driver import BaseDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.behavioral_cloning.model import BCPolicy


class BCDriver(BaseDriver):
    """Drives using a trained behavioral cloning policy."""

    def __init__(self, model_path: str | Path = "models/bc_v1.pth", stats_path: str | Path = "models/bc_v1.npz"):
        """Load trained BC model and normalization stats.

        Parameters
        ----------
        model_path : str | Path
            Path to saved BCPolicy weights
        stats_path : str | Path
            Path to saved normalization stats (mean, std)
        """
        self.model_path = Path(model_path)
        self.stats_path = Path(stats_path)
        self.model = None
        self.X_mean = None
        self.X_std = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """Load model and normalization stats."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats not found: {self.stats_path}")

        # Load stats
        stats = np.load(self.stats_path)
        self.X_mean = torch.from_numpy(stats["mean"]).float().to(self._device)
        self.X_std = torch.from_numpy(stats["std"]).float().to(self._device)

        # Load model
        self.model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(self._device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self._device))
        self.model.eval()

        print(f"BC model loaded from {self.model_path} (4 outputs: steer, accel, brake, gear)")

    def reset(self) -> None:
        """Reset driver state (no persistent state in BC)."""
        pass

    def on_restart(self) -> None:
        """Called when race restarts."""
        pass

    def step(self, state: SensorState) -> Action:
        """Inference: sensor state → action using trained BC policy."""
        # Build sensor vector: [angle, speed, speedY, speedZ, trackPos, track_0-18, rpm, gear]
        sensor_vec = np.array([
            state.angle,
            state.speed,
            state.speedY,
            state.speedZ,
            state.trackPos,
            *state.track,  # track_0 through track_18 (19 values)
            state.rpm,
            float(state.gear),
        ], dtype=np.float32)

        # Normalize
        sensor_tensor = torch.from_numpy(sensor_vec).float().to(self._device)
        sensor_tensor = (sensor_tensor - self.X_mean) / self.X_std

        # Forward pass
        with torch.no_grad():
            action_pred = self.model(sensor_tensor).cpu().numpy()

        steer, accel, brake, gear_pred = action_pred
        # Round gear to nearest integer (1-6)
        gear = int(round(float(gear_pred)))
        gear = max(1, min(6, gear))  # clamp to valid range

        return Action(steer=float(steer), accel=float(accel), brake=float(brake), gear=gear).clamp()
