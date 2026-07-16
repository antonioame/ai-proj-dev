"""Componenti condivisi fra i driver BC (_DRIVER/driver.py, drivers/bc_dagger/driver.py)
e il driver RL (drivers/rl/driver.py): l'architettura BCPolicy, il caricamento di
un checkpoint+statistiche di normalizzazione, la fase di avvio a marcia fissa, e il
cambio marcia basato su soglie RPM. Prima di questo modulo ciascun driver duplicava
questi ~40 righe verbatim — tenuti qui evita che le tre copie divergano silenziosamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Soglie di cambio marcia condivise da tutti i driver BC/RL
GEAR_UP_RPM = 12000.0
GEAR_DOWN_RPM = 6000.0

# Fase di avvio: accelerazione piena con sterzo a zero per questo numero di step.
# Mantiene l'auto dritta mentre il modello riceve input fuori distribuzione
# (OOD: velocità≈0, marcia=0).
STARTUP_STEPS = 80


class BCPolicy(nn.Module):
    """MLP di Behavioral Cloning per la guida in TORCS."""

    def __init__(self, input_dim: int = 26, hidden_dims: list | None = None):
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


def load_bc_model(model_path: Path, stats_path: Path, device: torch.device):
    """Carica un modello BCPolicy e le sue statistiche di normalizzazione."""
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"]).float().to(device)
    std = torch.from_numpy(stats["std"]).float().to(device)
    model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, mean, std


def startup_gear(speed: float) -> int:
    """Marcia da usare durante la fase di avvio, in base alla velocità corrente."""
    if speed < 15.0:
        return 1
    if speed < 45.0:
        return 2
    return 3


def shift_gear(rpm: float, current_gear: int) -> int:
    """Applica il cambio marcia basato su soglie RPM a current_gear."""
    if rpm > GEAR_UP_RPM and current_gear < 6:
        return current_gear + 1
    if rpm < GEAR_DOWN_RPM and current_gear > 1:
        return current_gear - 1
    return current_gear
