"""Driver RL residual — il driver BC funzionante più una correzione appresa e limitata.

Rispecchia per la guida dal vivo il ResidualTorcsSacEnv usato in fase di
training: l'azione di base arriva dal `_DRIVER.driver.BCDriver` completo, e la
policy SAC aggiunge un piccolo residuo (scalato da RESIDUAL_SCALE) su
steer/accel/brake. Stessa interfaccia step() di BCDriver/RLDriver, quindi
scripts/evaluate_rl.py --residual e scripts/run_agent_rl.py possono usarlo
come sostituto diretto.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from stable_baselines3 import SAC

from _DRIVER.driver import BCDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.features import build_feature_vector
from training.rl.residual_env import RESIDUAL_SCALE

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "sac_corkscrew_residual.zip"
NORM_STATS_PATH = Path(__file__).resolve().parents[2] / "_DRIVER" / "models" / "bc_from_olddriver_v1.npz"


class ResidualRLDriver:
    """Driver base BC + residuo SAC addestrato e limitato, deterministico in eval."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, residual_scale: float = RESIDUAL_SCALE):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Residual RL checkpoint not found: {checkpoint_path}. "
                "Train one with training/rl/train_sac.py --residual."
            )
        self._bc = BCDriver()
        self._model = SAC.load(str(checkpoint_path), device="cpu")
        self._res_scale = residual_scale

        stats = np.load(NORM_STATS_PATH)
        self._obs_mean = stats["mean"].astype(np.float32)
        self._obs_std = stats["std"].astype(np.float32)
        print(f"[ResidualRLDriver] BC base + SAC residual checkpoint: {checkpoint_path}")

    def reset(self) -> None:
        self._bc.reset()

    def on_restart(self) -> None:
        self._bc.on_restart()

    def step(self, state: SensorState) -> Action:
        base = self._bc.step(state)

        obs = (build_feature_vector(state) - self._obs_mean) / self._obs_std
        residual, _ = self._model.predict(obs, deterministic=True)

        return Action(
            steer=base.steer + float(residual[0]) * self._res_scale,
            accel=base.accel + float(residual[1]) * self._res_scale,
            brake=base.brake + float(residual[2]) * self._res_scale,
            gear=base.gear,
        ).clamp()
