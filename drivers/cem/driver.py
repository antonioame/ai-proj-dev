"""Driver per valutare un checkpoint ottimizzato con CEM (training/rl/train_cem.py).

Stessa interfaccia step()/on_restart() degli altri driver — sostituibile in
scripts/evaluate_cem.py senza toccare scripts/evaluate.py.

Struttura IDENTICA a _DRIVER/driver.py.BCDriver: due sotto-reti (rettilineo +
curva) fuse in base a track[9], non un singolo modello. Scoperto necessario
dopo che una CemPolicy a singola rete (solo pesi bc_from_olddriver_v1) è
risultata bloccata a 142,87s — 21s peggio della vera BC (121,978s) — proprio
perché le manca la specializzazione rettilineo/curva che il blend fornisce.

Ogni sotto-rete usa la PROPRIA normalizzazione (mean/std), non condivisa —
altro dettaglio di BCDriver facile da perdere: straight_model e corner_model
sono stati addestrati su dataset diversi con statistiche diverse. Usare la
normalizzazione sbagliata su una delle due reti produce un comportamento
completamente rotto (verificato: 82% fuori pista, danno 91, con i pesi BC
esatti — bug ora corretto registrando mean/std come buffer per sotto-rete).
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

MODELS_DIR = Path(__file__).resolve().parents[2] / "drivers" / "rl" / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "cem_v1.pth"
STRAIGHT_STATS_PATH = Path(__file__).resolve().parents[2] / "_DRIVER" / "models" / "bc_from_attempt1_v1.npz"
CORNER_STATS_PATH = Path(__file__).resolve().parents[2] / "_DRIVER" / "models" / "bc_from_olddriver_v1.npz"

_STEER_GAIN = 1.8
_ACCEL_GAIN = 1.40
_BRAKE_GAIN = 0.80
_STARTUP_STEPS = 80
_GEAR_UP_RPM = 12000.0
_GEAR_DOWN_RPM = 6000.0

_STRAIGHT_THRESHOLD = 44.0
_CORNER_THRESHOLD = 22.0


class SubPolicy(nn.Module):
    """Una delle due sotto-reti (rettilineo o curva), stessa architettura di
    _DRIVER/driver.py.BCPolicy, con la propria normalizzazione integrata
    (registrata come buffer — persiste nel checkpoint, CEM non la perturba
    perché non è un nn.Parameter)."""

    def __init__(self, obs_mean: np.ndarray, obs_std: np.ndarray, input_dim: int = 26, hidden_dims: list | None = None):
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
        self.register_buffer("obs_mean", torch.from_numpy(obs_mean.astype(np.float32)))
        self.register_buffer("obs_std", torch.from_numpy(obs_std.astype(np.float32)))

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        x = (raw_x - self.obs_mean) / self.obs_std
        features = self.backbone(x)
        steer = torch.tanh(self.head_steer(features))
        accel = torch.sigmoid(self.head_accel(features))
        brake = torch.sigmoid(self.head_brake(features))
        return torch.cat([steer, accel, brake], dim=-1)


class HybridCemPolicy(nn.Module):
    """Due SubPolicy (straight/corner) fuse esattamente come BCDriver._blend_weight."""

    def __init__(self):
        super().__init__()
        straight_stats = np.load(STRAIGHT_STATS_PATH)
        corner_stats = np.load(CORNER_STATS_PATH)
        self.straight = SubPolicy(straight_stats["mean"], straight_stats["std"])
        self.corner = SubPolicy(corner_stats["mean"], corner_stats["std"])

    def _blend_weight(self, front_dist: float) -> float:
        if front_dist >= _STRAIGHT_THRESHOLD:
            return 0.0
        if front_dist <= _CORNER_THRESHOLD:
            return 1.0
        return 1.0 - (front_dist - _CORNER_THRESHOLD) / (_STRAIGHT_THRESHOLD - _CORNER_THRESHOLD)

    def forward(self, raw_x: torch.Tensor, front_dist: float) -> torch.Tensor:
        w_corner = self._blend_weight(front_dist)
        w_straight = 1.0 - w_corner
        out_straight = self.straight(raw_x)
        out_corner = self.corner(raw_x)
        return w_straight * out_straight + w_corner * out_corner


class CemDriver:
    """Driver deterministico basato su un checkpoint ibrido ottimizzato con CEM."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"CEM checkpoint not found: {checkpoint_path}. Train one with training/rl/train_cem.py."
            )
        self._model = HybridCemPolicy()
        self._model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self._model.eval()

        self.current_gear = 1
        self._step_count = 0
        print(f"[CemDriver] Loaded checkpoint: {checkpoint_path}")

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

        if self._step_count <= _STARTUP_STEPS:
            gear = self._startup_gear(state.speed)
            self.current_gear = gear
            return Action(steer=0.0, accel=1.0, brake=0.0, gear=gear).clamp()

        from training.rl.features import build_feature_vector

        raw = build_feature_vector(state).astype(np.float32)
        front_dist = state.track[9] if len(state.track) > 9 else 100.0
        with torch.no_grad():
            out = self._model(torch.from_numpy(raw), front_dist).numpy()
        steer, accel, brake = out.tolist()

        if state.rpm > _GEAR_UP_RPM and self.current_gear < 6:
            self.current_gear += 1
        elif state.rpm < _GEAR_DOWN_RPM and self.current_gear > 1:
            self.current_gear -= 1

        return Action(
            steer=steer * _STEER_GAIN,
            accel=accel * _ACCEL_GAIN,
            brake=brake * _BRAKE_GAIN,
            gear=self.current_gear,
        ).clamp()
