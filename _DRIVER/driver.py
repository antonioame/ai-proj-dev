"""Driver di Behavioral Cloning — modello singolo clonato dallo stile di guida
del bot nativo TORCS "tita" (bc_tita_v20), promosso a driver di produzione il
2026-07-15 al posto del precedente blend rettilineo/curva (bc_from_attempt1_v1
+ bc_from_olddriver_v1), verificato più lento (124.296s vs 111.986s, stesse
condizioni di test, entrambi puliti: 0% fuori pista, 0 danni).

I due modelli del vecchio blend restano in models/ (bc_from_attempt1_v1.*,
bc_from_olddriver_v1.*) per eventuale rollback, semplicemente non più caricati
da questa classe.
"""

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
    """Driver BC a modello singolo, clonato dallo stile di guida di tita
    (bc_tita_v20: 13 giri puliti di telemetria del bot nativo + esempi di
    recupero raccolti con procedura DAgger-style, bc come rete di sicurezza).

    A differenza del precedente blend rettilineo/curva a due reti, questo è
    un unico BCPolicy che gestisce l'intero giro.
    """

    STEER_GAIN = 1.0   # Applicato all'output di sterzo — NON 1.8 (valore del
                        # vecchio blend): con questo modello, un gain più alto
                        # causava oscillazioni e uscite di pista in curva.
    ACCEL_GAIN = 1.40
    BRAKE_GAIN = 0.80

    # Fase di avvio: applica pieno gas con sterzo a zero per questo numero di step.
    # Mantiene l'auto dritta mentre il modello riceve input fuori distribuzione
    # (OOD: velocità≈0, marcia=0).
    STARTUP_STEPS = 80

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_gear = 1
        self._step_count = 0

        models_dir = Path(__file__).resolve().parent / "models"
        model_path = models_dir / "bc_tita_v20.pth"
        stats_path = models_dir / "bc_tita_v20.npz"

        for p in [model_path, stats_path]:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")

        self._model, self._mean, self._std = _load_bc_model(model_path, stats_path, self._device)
        print("[BCDriver] Loaded checkpoint: bc_tita_v20 (clone dello stile di guida di tita)")

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
