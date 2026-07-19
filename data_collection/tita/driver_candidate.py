"""Driver BC a modello singolo per valutare il checkpoint candidato bc_tita_v1
(addestrato ESCLUSIVAMENTE su telemetria tita, data_collection/tita/train_tita_only.py).

Stessa architettura/interfaccia di drivers/bc_dagger/driver.py (BCDaggerDriver)
e stessa logica di avvio/cambio marcia; i gain pero' NON sono identici:
STEER_GAIN qui e' 1.0 (non 1.8 come in BCDaggerDriver): differenza
intenzionale, verificata empiricamente: col modello clonato da tita un gain
di sterzo alto causava oscillazioni e uscite di pista in curva. Unica altra
differenza e' il checkpoint caricato (data_collection/tita/candidate_models/,
non _DRIVER/models/ ne' drivers/).

Non tocca _DRIVER/, drivers/, training/ ne' gli script BC esistenti.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from drivers.bc_common import BCPolicy, shift_gear, startup_gear
from torcs_env.actions import Action
from torcs_env.sensors import SensorState

CANDIDATE_DIR = Path(__file__).resolve().parent / "candidate_models"
DEFAULT_CHECKPOINT = CANDIDATE_DIR / "bc_tita_v1.pth"
DEFAULT_STATS = CANDIDATE_DIR / "bc_tita_v1.npz"


class BCTitaCandidateDriver:
    """Driver BC a modello singolo, addestrato solo su telemetria tita."""

    STEER_GAIN = 1.0
    ACCEL_GAIN = 1.40
    BRAKE_GAIN = 0.80
    STARTUP_STEPS = 80

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, stats_path: Path = DEFAULT_STATS):
        if not checkpoint_path.exists() or not stats_path.exists():
            raise FileNotFoundError(
                f"bc_tita candidate checkpoint not found: {checkpoint_path} / {stats_path}. "
                "Train one with data_collection/tita/train_tita_only.py."
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
        print(f"[BCTitaCandidateDriver] Loaded checkpoint: {checkpoint_path}")

    def reset(self) -> None:
        pass

    def on_restart(self) -> None:
        self.current_gear = 1
        self._step_count = 0

    def step(self, state: SensorState) -> Action:
        self._step_count += 1

        if self._step_count <= self.STARTUP_STEPS:
            gear = startup_gear(state.speed)
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

        self.current_gear = shift_gear(state.rpm, self.current_gear)

        return Action(
            steer=float(steer),
            accel=float(accel),
            brake=float(brake),
            gear=self.current_gear,
        ).clamp()
