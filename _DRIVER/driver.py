"""Driver di Behavioral Cloning — ibrido di due modelli BC fusi in base al contesto della pista."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from torcs_env.actions import Action
from torcs_env.sensors import SensorState


class BCPolicy(nn.Module):
    """MLP di Behavioral Cloning per la guida in TORCS."""
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


def _load_bc_model(model_path: Path, stats_path: Path, device: torch.device):
    """Carica un modello BCPolicy e le sue statistiche di normalizzazione."""
    stats = np.load(stats_path)
    X_mean = torch.from_numpy(stats["mean"]).float().to(device)
    X_std = torch.from_numpy(stats["std"]).float().to(device)
    model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, X_mean, X_std


class BCDriver:
    """Driver BC ibrido: fonde due modelli in base al contesto rettilineo/curva.

    - straight_model: addestrato sulla telemetria del precedente tentativo di
      driving-net (bc_source_driver/), a sua volta addestrato su dati rule_based
      — migliore sui rettilinei
    - corner_model:   addestrato sulla telemetria del vecchio driver ibrido
      (project_V1, regole + predittore BC) — migliore in curva

    Il peso della fusione è determinato da track[9] (sensore di distanza frontale):
      - track[9] > STRAIGHT_THRESHOLD → modello rettilineo puro
      - track[9] < CORNER_THRESHOLD   → modello curva puro
      - tra i due                     → fusione lineare morbida
    """

    STEER_GAIN = 1.8   # Applicato all'output di sterzo fuso
    ACCEL_GAIN = 1.40  # Applicato all'output di accelerazione fuso
    BRAKE_GAIN = 0.80  # Applicato all'output di frenata fuso

    STRAIGHT_THRESHOLD = 44.0   # m — sopra questa soglia: modello rettilineo puro
    CORNER_THRESHOLD   = 22.0   # m — sotto questa soglia: modello curva puro

    # Fase di avvio: applica pieno gas con sterzo a zero per questo numero di step.
    # Mantiene l'auto dritta mentre i modelli ricevono input fuori distribuzione
    # (OOD: velocità≈0, marcia=0).
    STARTUP_STEPS = 80

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_gear = 1
        self._step_count = 0

        models_dir = Path(__file__).resolve().parent / "models"
        straight_model_path = models_dir / "bc_from_attempt1_v1.pth"
        straight_stats_path = models_dir / "bc_from_attempt1_v1.npz"
        corner_model_path   = models_dir / "bc_from_olddriver_v1.pth"
        corner_stats_path   = models_dir / "bc_from_olddriver_v1.npz"

        for p in [straight_model_path, straight_stats_path, corner_model_path, corner_stats_path]:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")

        self.straight_model, self.straight_mean, self.straight_std = _load_bc_model(
            straight_model_path, straight_stats_path, self._device
        )
        self.corner_model, self.corner_mean, self.corner_std = _load_bc_model(
            corner_model_path, corner_stats_path, self._device
        )
        print("[BCDriver] Hybrid model loaded: straight=bc_from_attempt1_v1, corner=bc_from_olddriver_v1")

    def reset(self) -> None:
        pass

    def on_restart(self) -> None:
        self.current_gear = 1
        self._step_count = 0

    def _infer(self, model, X_mean, X_std, sensor_vec: np.ndarray) -> dict:
        """Esegue l'inferenza su un singolo modello."""
        t = torch.from_numpy(sensor_vec).float().to(self._device)
        t = (t - X_mean) / X_std
        with torch.no_grad():
            out = model(t)
        return {k: float(v.squeeze().item()) for k, v in out.items()}

    def _blend_weight(self, front_dist: float) -> float:
        """Restituisce il peso curva in [0, 1]. 0 = rettilineo puro, 1 = curva pura."""
        if front_dist >= self.STRAIGHT_THRESHOLD:
            return 0.0
        if front_dist <= self.CORNER_THRESHOLD:
            return 1.0
        return 1.0 - (front_dist - self.CORNER_THRESHOLD) / (self.STRAIGHT_THRESHOLD - self.CORNER_THRESHOLD)

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

        # Inferenza da entrambi i modelli
        straight_out = self._infer(self.straight_model, self.straight_mean, self.straight_std, sensor_vec)
        corner_out   = self._infer(self.corner_model,   self.corner_mean,   self.corner_std,   sensor_vec)

        # Fusione basata sulla distanza del sensore frontale
        front_dist = state.track[9] if len(state.track) > 9 else 100.0
        w_corner = self._blend_weight(front_dist)
        w_straight = 1.0 - w_corner

        steer = w_straight * straight_out["steer"] + w_corner * corner_out["steer"]
        accel = w_straight * straight_out["accel"] + w_corner * corner_out["accel"]
        brake = w_straight * straight_out["brake"] + w_corner * corner_out["brake"]

        # Applica i guadagni
        steer = max(-1.0, min(1.0, steer * self.STEER_GAIN))
        accel = max(0.0,  min(1.0, accel * self.ACCEL_GAIN))
        brake = max(0.0,  min(1.0, brake * self.BRAKE_GAIN))

        # Gestione marcia basata su RPM
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
