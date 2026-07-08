"""Phase 3 RL driver — SAC fine-tuned from the BC corner model.

Same step() interface as _DRIVER.driver.BCDriver so it's a drop-in swap for
scripts/run_agent_rl.py / scripts/evaluate_rl.py. Not the active/default
driver — see CLAUDE.md Phase 3 section for promotion criteria (must match or
beat the BC baseline on safety and not be worse on lap time).
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

# Mirrors _DRIVER/driver.py.BCDriver so the RL driver's launch/gear behaviour
# matches what the policy was actually trained under (training/rl/torcs_gym_env.py).
_GEAR_UP_RPM = 12000.0
_GEAR_DOWN_RPM = 6000.0
_STARTUP_STEPS = 80


class RLDriver:
    """Loads a trained SAC checkpoint and drives with deterministic actions."""

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

        return Action(steer=steer, accel=accel, brake=brake, gear=self.current_gear).clamp()
