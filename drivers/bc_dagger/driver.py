"""Driver BC a modello singolo per valutare il checkpoint bc_dagger_v1.

Carica un unico modello BCPolicy addestrato su dataset originale + DAgger
filtrato (scripts/train/train_bc_dagger.py). Stessa interfaccia step()/
on_restart() degli altri driver, così scripts/eval/evaluate_bc_dagger.py può
sostituirlo direttamente senza toccare scripts/eval/evaluate.py.

Guadagni post-hoc e logica di avvio/cambio marcia identici al BCDriver
dell'epoca in cui bc_dagger_v1 è stato addestrato (il blend pre-2026-07-15:
STEER_GAIN=1.8/ACCEL=1.40/BRAKE=0.80/STARTUP_STEPS=80), per un confronto equo
a parità di pipeline di controllo. Nota: il BCDriver di produzione attuale
(bc_tita_v20, anch'esso a modello singolo) usa STEER_GAIN=1.0 — i valori qui
NON vanno allineati a quello, sono legati a questo checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from drivers.bc_common import load_bc_model, shift_gear, startup_gear
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.features import build_feature_vector

MODELS_DIR = Path(__file__).resolve().parents[2] / "_DRIVER" / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "bc_dagger_v1.pth"
DEFAULT_STATS = MODELS_DIR / "bc_dagger_v1.npz"


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
                "Train one with scripts/train/train_bc_dagger.py."
            )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_gear = 1
        self._step_count = 0

        self._model, self._mean, self._std = load_bc_model(checkpoint_path, stats_path, self._device)
        print(f"[BCDaggerDriver] Loaded checkpoint: {checkpoint_path}")

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

        sensor_vec = build_feature_vector(state)

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
