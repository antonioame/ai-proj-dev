"""Blend BC legacy (rettilineo/curva) — base congelata su cui è stato addestrato
il checkpoint residual consegnato.

Perché esiste
-------------
`drivers/rl/models/sac_corkscrew_residual.zip` (run 20260710_025418, eval
documentata 127.07s / 0% off-track) è stato addestrato quando
`_DRIVER.driver.BCDriver` era ANCORA il blend a due reti (bc_from_attempt1_v1
rettilineo + bc_from_olddriver_v1 curva, STEER_GAIN=1.8/ACCEL=1.40/BRAKE=0.80,
soglie 44/22 m su track[9], 121.978s — commit `62fe930`). Il 2026-07-15
(commit `0d3c7b8`) `_DRIVER.driver.BCDriver` è stato sostituito dal modello
singolo `bc_tita_v20` (111.986s, STEER_GAIN=1.0): il residual, se sommato a
quella base nuova, corregge un comportamento diverso da quello su cui è stato
addestrato — mai validato.

Questa classe replica esattamente il vecchio `BCDriver` così com'era
nell'ultimo commit prima della promozione (`git show 62fe930:_DRIVER/driver.py`),
per ridare al residual la base su cui è stato effettivamente addestrato. NON è
il driver di produzione (quello, dal 2026-07-19, è `_DRIVER.driver.BCDriver` =
cem_v5, dopo bc_tita_v20 dal 2026-07-15 al 2026-07-19) e non va evoluta: è
congelata di proposito, esiste solo per rendere riproducibile il checkpoint
residual esistente.

I pesi (`_DRIVER/models/bc_from_attempt1_v1.*`, `bc_from_olddriver_v1.*`) sono
tenuti nel repo apposta per questo rollback.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from drivers.bc_common import STARTUP_STEPS, load_bc_model, shift_gear, startup_gear
from torcs_env.actions import Action
from torcs_env.sensors import SensorState


class LegacyBlendBCDriver:
    """Driver BC ibrido legacy: fonde due modelli in base al contesto pista.

    - straight_model: bc_from_attempt1_v1 — migliore in rettilineo
    - corner_model:   bc_from_olddriver_v1 — migliore in curva

    Il peso di blend dipende da track[9] (sensore di distanza frontale):
      - track[9] > STRAIGHT_THRESHOLD → modello rettilineo puro
      - track[9] < CORNER_THRESHOLD   → modello curva puro
      - fra i due                     → blend lineare
    """

    STEER_GAIN = 1.8
    ACCEL_GAIN = 1.40
    BRAKE_GAIN = 0.80

    STRAIGHT_THRESHOLD = 44.0  # m — sopra: modello rettilineo puro
    CORNER_THRESHOLD = 22.0  # m — sotto: modello curva puro

    STARTUP_STEPS = STARTUP_STEPS

    def __init__(self):
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_gear = 1
        self._step_count = 0

        models_dir = Path(__file__).resolve().parents[2] / "_DRIVER" / "models"
        straight_model_path = models_dir / "bc_from_attempt1_v1.pth"
        straight_stats_path = models_dir / "bc_from_attempt1_v1.npz"
        corner_model_path = models_dir / "bc_from_olddriver_v1.pth"
        corner_stats_path = models_dir / "bc_from_olddriver_v1.npz"

        for p in [straight_model_path, straight_stats_path, corner_model_path, corner_stats_path]:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")

        self.straight_model, self.straight_mean, self.straight_std = load_bc_model(
            straight_model_path, straight_stats_path, self._device
        )
        self.corner_model, self.corner_mean, self.corner_std = load_bc_model(
            corner_model_path, corner_stats_path, self._device
        )
        print("[LegacyBlendBCDriver] Blend legacy caricato: straight=bc_from_attempt1_v1, corner=bc_from_olddriver_v1")

    def reset(self) -> None:
        pass

    def on_restart(self) -> None:
        self.current_gear = 1
        self._step_count = 0

    def _infer(self, model, X_mean, X_std, sensor_vec: np.ndarray) -> dict:
        import torch

        t = torch.from_numpy(sensor_vec).float().to(self._device)
        t = (t - X_mean) / X_std
        with torch.no_grad():
            out = model(t)
        return {k: float(v.squeeze().item()) for k, v in out.items()}

    def _blend_weight(self, front_dist: float) -> float:
        """Peso curva in [0, 1]. 0 = rettilineo puro, 1 = curva pura."""
        if front_dist >= self.STRAIGHT_THRESHOLD:
            return 0.0
        if front_dist <= self.CORNER_THRESHOLD:
            return 1.0
        return 1.0 - (front_dist - self.CORNER_THRESHOLD) / (self.STRAIGHT_THRESHOLD - self.CORNER_THRESHOLD)

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

        straight_out = self._infer(self.straight_model, self.straight_mean, self.straight_std, sensor_vec)
        corner_out = self._infer(self.corner_model, self.corner_mean, self.corner_std, sensor_vec)

        front_dist = state.track[9] if len(state.track) > 9 else 100.0
        w_corner = self._blend_weight(front_dist)
        w_straight = 1.0 - w_corner

        steer = w_straight * straight_out["steer"] + w_corner * corner_out["steer"]
        accel = w_straight * straight_out["accel"] + w_corner * corner_out["accel"]
        brake = w_straight * straight_out["brake"] + w_corner * corner_out["brake"]

        steer = max(-1.0, min(1.0, steer * self.STEER_GAIN))
        accel = max(0.0, min(1.0, accel * self.ACCEL_GAIN))
        brake = max(0.0, min(1.0, brake * self.BRAKE_GAIN))

        self.current_gear = shift_gear(state.rpm, self.current_gear)

        return Action(
            steer=float(steer),
            accel=float(accel),
            brake=float(brake),
            gear=self.current_gear,
        ).clamp()
