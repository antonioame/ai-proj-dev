"""Driver RL Fase 3 — SAC fine-tuned a partire dal modello BC per le curve.

Stessa interfaccia step() di _DRIVER.driver.BCDriver, quindi è intercambiabile
in scripts/run_agent_rl.py / scripts/evaluate_rl.py. Non è il driver
attivo/di default: SAC puro (senza base BC) sfrutta tutta l'autorità sul
reward di velocità e va in reward-hacking — con i checkpoint disponibili
(sac_corkscrew_v1, sac_corkscrew_refined_v2) la policy si blocca (0 giri
completati, velocità media <1 km/h — vedi la sezione Fase 3 di CLAUDE.md).
Il driver RL funzionante è drivers/rl/residual_driver.py (ResidualRLDriver).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from stable_baselines3 import SAC

from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.features import build_feature_vector

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "sac_corkscrew_v1.zip"
NORM_STATS_PATH = Path(__file__).resolve().parents[2] / "_DRIVER" / "models" / "bc_from_olddriver_v1.npz"

# Rispecchia _DRIVER/driver.py.BCDriver in modo che il comportamento di
# avvio/marce del driver RL corrisponda a ciò su cui la policy è stata
# effettivamente addestrata (training/rl/torcs_gym_env.py).
_GEAR_UP_RPM = 12000.0
_GEAR_DOWN_RPM = 6000.0
_STARTUP_STEPS = 80

# Stessi guadagni post-hoc di torcs_gym_env.py/BCDriver — devono combaciare
# esattamente con quelli applicati in training, altrimenti l'inferenza non
# rispecchia ciò su cui la policy è stata addestrata.
_STEER_GAIN = 1.8
_ACCEL_GAIN = 1.40
_BRAKE_GAIN = 0.80


class RLDriver:
    """Carica un checkpoint SAC addestrato e guida con azioni deterministiche."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"RL checkpoint not found: {checkpoint_path}. "
                "Train one first with training/rl/train_sac.py."
            )
        self._model = SAC.load(str(checkpoint_path), device="cpu")

        stats = np.load(NORM_STATS_PATH)
        self._obs_mean = stats["mean"].astype(np.float32)
        self._obs_std = stats["std"].astype(np.float32)

        self.current_gear = 1
        self._step_count = 0
        print(f"[RLDriver] Loaded SAC checkpoint: {checkpoint_path}")

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

        raw = build_feature_vector(state)
        obs = (raw - self._obs_mean) / self._obs_std

        action, _ = self._model.predict(obs, deterministic=True)
        steer, accel, brake = (float(a) for a in action)

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
