"""Driver BC a modello singolo per valutare il checkpoint bc_dagger_v1.

A differenza di _DRIVER.driver.BCDriver (blend rettilineo/curva di due reti),
questo carica un unico modello BCPolicy addestrato su dataset originale +
DAgger filtrato (scripts/train_bc_dagger.py). Stessa interfaccia step()/
on_restart() degli altri driver, così scripts/evaluate_bc_dagger.py può
sostituirlo direttamente senza toccare scripts/evaluate.py.

Guadagni post-hoc e logica di avvio/cambio marcia identici a BCDriver (stessi
valori STEER_GAIN/ACCEL_GAIN/BRAKE_GAIN/STARTUP_STEPS), per un confronto equo
tra checkpoint a parità di tutto il resto della pipeline di controllo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn

from torcs_env.actions import Action
from torcs_env.sensors import SensorState

MODELS_DIR = Path(__file__).resolve().parents[2] / "_DRIVER" / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "bc_dagger_v1.pth"
DEFAULT_STATS = MODELS_DIR / "bc_dagger_v1.npz"


class BCPolicy(nn.Module):
    def __init__(self, input_dim: int = 26, hidden_dims: list | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU()]
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


class BCDaggerDriver:
    """Driver BC a modello singolo, addestrato su dataset originale + DAgger."""

    STEER_GAIN = 1.8
    ACCEL_GAIN = 1.40
    BRAKE_GAIN = 0.80
    STARTUP_STEPS = 80

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, stats_path: Path = DEFAULT_STATS):
        if not checkpoint_path.exists() or not stats_path.exists():
            raise FileNotFoundError(
                f"bc_dagger checkpoint not found: {checkpoint_path} / {stats_path}. "
                "Train one with scripts/train_bc_dagger.py."
            )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_gear = 1
        self._step_count = 0

        stats = np.load(stats_path)
        self._mean = torch.from_numpy(stats["mean"]).float().to(self._device)
        self._std = torch.from_numpy(stats["std"]).float().to(self._device)

        self._model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(self._device)
        self._model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        self._model.eval()
        print(f"[BCDaggerDriver] Loaded checkpoint: {checkpoint_path}")

    def reset(self) -> None:
        pass

    def on_restart(self) -> None:
        self.current_gear = 1
        self._step_count = 0

    def _startup_gear(self, speed: float) -> int:
        if speed < 15.0:
            return 1
        if speed < 45.0:
            return 2
        return 3

    def step(self, state: SensorState) -> Action:
        self._step_count += 1

        if self._step_count <= self.STARTUP_STEPS:
            gear = self._startup_gear(state.speed)
            self.current_gear = gear
            return Action(steer=0.0, accel=1.0, brake=0.0, gear=gear).clamp()

        sensor_vec = np.array([
            state.angle,
            state.speed,
            state.speedY,
            state.speedZ,
            state.trackPos,
            *state.track,
            state.rpm,
            float(state.gear),
        ], dtype=np.float32)

        t = torch.from_numpy(sensor_vec).float().to(self._device)
        t = (t - self._mean) / self._std
        with torch.no_grad():
            out = self._model(t)

        steer = max(-1.0, min(1.0, float(out["steer"].item()) * self.STEER_GAIN))
        accel = max(0.0, min(1.0, float(out["accel"].item()) * self.ACCEL_GAIN))
        brake = max(0.0, min(1.0, float(out["brake"].item()) * self.BRAKE_GAIN))

        if state.rpm > 12000 and self.current_gear < 6:
            self.current_gear += 1
        elif state.rpm < 6000 and self.current_gear > 1:
            self.current_gear -= 1

        return Action(
            steer=float(steer),
            accel=float(accel),
            brake=float(brake),
            gear=self.current_gear,
        ).clamp()
