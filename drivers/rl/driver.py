"""RL driver: wraps a Stable-Baselines3 checkpoint (DDPG or PPO) for live driving.

The model is loaded in a background thread so the TORCS handshake completes
before the SCR pre-connection timeout fires. While loading, the car drives
gently straight (same fallback pattern as BCDriver).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from drivers.base_driver import BaseDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.gym_env import (
    _TRACK_IDX,
    _GEAR_UP_RPM,
    _GEAR_DOWN_RPM,
    _GEAR_COOLDOWN,
    _OBS_MEAN,
    _OBS_STD,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "ddpg_v1"


class RLDriver(BaseDriver):
    """Drive with a trained SB3 policy (DDPG or PPO).

    Parameters
    ----------
    model_path:
        Path to the saved SB3 model (with or without the `.zip` extension).
    algo:
        ``"ddpg"`` (default) or ``"ppo"``.
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL,
        algo: str = "ddpg",
    ) -> None:
        self._model_path = Path(model_path)
        self._algo = algo.lower()
        self._model = None
        self._loaded = threading.Event()

        self._gear: int = 1
        self._step_count: int = 0
        self._last_gear_step: int = -_GEAR_COOLDOWN

        threading.Thread(target=self._load, daemon=True).start()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            if self._algo == "ppo":
                from stable_baselines3 import PPO as SB3Algo
            else:
                from stable_baselines3 import DDPG as SB3Algo

            model = SB3Algo.load(str(self._model_path), device="cpu")
            model.policy.set_training_mode(False)
            self._model = model
            self._loaded.set()
            logger.info(
                "RLDriver: %s loaded from %s", self._algo.upper(), self._model_path
            )
        except Exception as exc:
            logger.error("RLDriver: failed to load model — %s", exc)

    # ------------------------------------------------------------------
    # Observation builder (must match training/rl/gym_env.py exactly)
    # ------------------------------------------------------------------

    def _make_obs(self, state: SensorState) -> np.ndarray:
        raw = np.array(
            [
                state.speed,
                state.trackPos,
                state.angle,
                state.rpm,
                float(self._gear),
                state.track[_TRACK_IDX[0]],
                state.track[_TRACK_IDX[1]],
                state.track[_TRACK_IDX[2]],
            ],
            dtype=np.float32,
        )
        return (raw - _OBS_MEAN) / _OBS_STD

    # ------------------------------------------------------------------
    # Gear management (mirrors gym_env logic)
    # ------------------------------------------------------------------

    def _update_gear(self, state: SensorState) -> None:
        if (self._step_count - self._last_gear_step) < _GEAR_COOLDOWN:
            return
        if state.rpm > _GEAR_UP_RPM and self._gear < 6:
            self._gear += 1
            self._last_gear_step = self._step_count
        elif state.rpm < _GEAR_DOWN_RPM and self._gear > 1:
            self._gear -= 1
            self._last_gear_step = self._step_count

    # ------------------------------------------------------------------
    # BaseDriver interface
    # ------------------------------------------------------------------

    def step(self, state: SensorState) -> Action:
        self._step_count += 1
        self._update_gear(state)

        if not self._loaded.is_set():
            return Action(accel=0.3, steer=0.0, brake=0.0, gear=1)

        obs = self._make_obs(state)
        action, _ = self._model.predict(obs, deterministic=True)

        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        return Action(steer=steer, accel=accel, brake=brake, gear=self._gear).clamp()

    def on_restart(self) -> None:
        self._gear = 1
        self._step_count = 0
        self._last_gear_step = -_GEAR_COOLDOWN

    def reset(self) -> None:
        self.on_restart()
