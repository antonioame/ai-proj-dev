"""Behavioral Cloning driver: loads a trained MLPPolicy checkpoint and drives."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from drivers.base_driver import BaseDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "bc_v2.pth"


class BCDriver(BaseDriver):
    """MLP driver trained via behavioral cloning.

    torch and MLPPolicy are imported lazily inside the background thread so
    this module loads in <50 ms. The TORCS handshake therefore completes before
    the SCR pre-connection timeout (~2-3 s) fires.  While the checkpoint loads,
    step() returns a neutral action to satisfy the per-action timeout.
    """

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL) -> None:
        self._model_path = Path(model_path)
        self._model = None
        self._mean = None
        self._std = None
        self._loaded = threading.Event()
        self._current_gear = 1
        self._last_gear_change_step = -100
        self._step_count = 0
        # Auto gear thresholds (RPM)
        self._gear_up_rpm = 7500
        self._gear_down_rpm = 2500
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        import torch  # lazy: keeps module-level import fast
        from training.behavioral_cloning.model import MLPPolicy  # lazy: same reason

        try:
            ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
            model = MLPPolicy(
                input_dim=ckpt["input_dim"],
                hidden_dims=ckpt["hidden_dims"],
            )
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            mean = torch.from_numpy(ckpt["sensor_mean"].astype(np.float32))
            std = torch.from_numpy(ckpt["sensor_std"].astype(np.float32))
            # Write all fields before setting the event so step() never sees partial state.
            self._mean = mean
            self._std = std
            self._model = model
            self._loaded.set()
            logger.info("BCDriver: checkpoint loaded from %s", self._model_path)
        except Exception as e:
            logger.error("BCDriver: failed to load checkpoint: %s", e, exc_info=True)
            self._loaded.set()  # Set event anyway to unblock step()

    def step(self, state: SensorState) -> Action:
        self._step_count += 1
        if not self._loaded.is_set():
            # Drive gently straight while checkpoint loads in background.
            if self._step_count == 1:
                logger.info("BCDriver: waiting for model load...")
            return Action(accel=0.3, steer=0.0, brake=0.0, gear=1).clamp()
        if self._step_count == 51:  # Log once after loading complete
            logger.info("BCDriver: model loaded, starting inference")
        return self._infer(state)

    def _infer(self, state: SensorState) -> Action:
        import torch  # already in sys.modules once _load() has completed

        if self._model is None:
            logger.error("BCDriver: model is None, cannot infer")
            return Action(accel=0.3, steer=0.0, brake=0.0, gear=1).clamp()

        # Feature order must match SENSOR_COLS in dataset.py:
        # ["speed", "trackPos", "angle", "rpm", "gear", "track_6", "track_12", "track_18"]
        # track_6, track_12, track_18 are lookahead sensors at +6°, 0°, -6° angles
        x = torch.tensor(
            [state.speed, state.trackPos, state.angle, state.rpm, self._current_gear,
             state.track[6], state.track[12], state.track[18]],
            dtype=torch.float32,
        )
        x = (x - self._mean) / self._std
        x = x.unsqueeze(0)

        with torch.no_grad():
            out = self._model.predict(x)

        # Auto gear management with cooldown (min 5 steps ~250ms between changes)
        if (self._step_count - self._last_gear_change_step) >= 5:
            if state.rpm > self._gear_up_rpm and self._current_gear < 6:
                self._current_gear += 1
                self._last_gear_change_step = self._step_count
            elif state.rpm < self._gear_down_rpm and self._current_gear > 1:
                self._current_gear -= 1
                self._last_gear_change_step = self._step_count

        return Action(
            steer=float(out["steer"].item()),
            accel=float(out["accel"].item()),
            brake=float(out["brake"].item()),
            gear=self._current_gear,
        ).clamp()

    def on_restart(self) -> None:
        pass

    def reset(self) -> None:
        pass
