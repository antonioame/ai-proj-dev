"""Behavioral Cloning driver — runs trained BC policy on TORCS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn

from drivers.base_driver import BaseDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState


class BCPolicy(nn.Module):
    """Behavioral Cloning MLP for TORCS driving."""
    def __init__(self, input_dim: int = 26, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head_steer = nn.Linear(prev_dim, 1)
        self.head_accel = nn.Linear(prev_dim, 1)
        self.head_brake = nn.Linear(prev_dim, 1)
        self.head_gear = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "steer": torch.tanh(self.head_steer(features)),
            "accel": torch.sigmoid(self.head_accel(features)),
            "brake": torch.sigmoid(self.head_brake(features)),
            "gear": self.head_gear(features),
        }


class BCDriver(BaseDriver):
    """Drives using a trained behavioral cloning policy."""

    STEER_GAIN = 1.8  # Amplify steering to force tighter turns and inner-track positioning
    ACCEL_GAIN = 1.25  # Amplify acceleration to maintain higher speeds through curves

    def __init__(self, model_path: str | Path = "models/bc_from_olddriver_v1.pth", stats_path: str | Path = "models/bc_from_olddriver_v1.npz"):
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
        self.current_gear = 1
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

        print(f"BC model loaded from {self.model_path} (3 outputs: steer, accel, brake; gear managed separately)")

    def reset(self) -> None:
        """Reset driver state (no persistent state in BC)."""
        pass

    def on_restart(self) -> None:
        """Called when race restarts."""
        pass

    def step(self, state: SensorState) -> Action:
        """Inference: sensor state → action using trained BC policy.

        Gear management is handled separately (not predicted by model).
        """
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

        # Forward pass — model outputs dict: {steer, accel, brake, gear}
        # We use steer, accel, brake; gear is managed separately via RPM heuristic
        with torch.no_grad():
            action_pred = self.model(sensor_tensor)

        steer = float(action_pred["steer"].squeeze().item())
        accel = float(action_pred["accel"].squeeze().item())
        brake = float(action_pred["brake"].squeeze().item())

        # Amplify steering to follow tighter racing line
        steer = max(-1.0, min(1.0, steer * self.STEER_GAIN))
        # Amplify acceleration to maintain speed through curves
        accel = max(0.0, min(1.0, accel * self.ACCEL_GAIN))

        # Gear management: RPM-based upshift/downshift
        upshift_rpm = 12000
        downshift_rpm = 6000

        if state.rpm > upshift_rpm and self.current_gear < 6:
            self.current_gear += 1
        elif state.rpm < downshift_rpm and self.current_gear > 1:
            self.current_gear -= 1

        return Action(
            steer=float(steer),
            accel=float(accel),
            brake=float(brake),
            gear=self.current_gear
        ).clamp()
