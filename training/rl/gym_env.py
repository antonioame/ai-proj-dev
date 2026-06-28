"""Gymnasium environment wrapping TORCSClient for RL training (Phase 3).

Observation space (9 features):
  [0] speed       — forward speed, normalised by 300 km/h
  [1] trackPos    — lateral position on track (−1=left edge, +1=right edge)
  [2] angle       — car orientation vs track axis, normalised by π
  [3] rpm         — engine RPM, normalised by 10 000
  [4] gear        — current gear (1–6), normalised by 6
  [5] damage      — cumulative damage, normalised by 10 000
  [6] track[7]    — range-finder at −6°, normalised by 200 m
  [7] track[9]    — range-finder at   0° (dead ahead), normalised by 200 m
  [8] track[11]   — range-finder at  +6°, normalised by 200 m

Action space (3 continuous):
  [0] steer  ∈ [−1, 1]   (−1 = full right, +1 = full left)
  [1] accel  ∈ [ 0, 1]
  [2] brake  ∈ [ 0, 1]

Gear is handled by rule-based logic inside the env to keep the action space small.
"""

from __future__ import annotations

import logging
import numpy as np
import gymnasium as gym

from torcs_env.client import TORCSClient, RESTART, SHUTDOWN
from torcs_env.sensors import SensorState
from torcs_env.actions import Action
from training.rl.reward import compute_reward

logger = logging.getLogger(__name__)

# Indices into the 19-element track sensor array that we include in obs
_TRACK_IDX = (7, 9, 11)   # ≈ −6°, 0°, +6°
OBS_DIM = 9

_GEAR_UP_RPM = 9000
_GEAR_DOWN_RPM = 3500
_GEAR_COOLDOWN = 5  # min steps between gear changes


class TORCSGymEnv(gym.Env):
    """Single-instance TORCS gymnasium environment.

    Each call to reset() sends (meta 1) to restart the race from a standing
    start.  Only one env instance per TORCS UDP port is supported; for parallel
    training start multiple TORCS processes on different ports and wrap them with
    SB3's SubprocVecEnv.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        max_steps: int = 10_000,
        stuck_speed_kmh: float = 5.0,
        stuck_patience: int = 300,
    ) -> None:
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._host = host
        self._port = port
        self._max_steps = max_steps
        self._stuck_speed = stuck_speed_kmh
        self._stuck_patience = stuck_patience

        self._client: TORCSClient | None = None
        self._prev_state: SensorState | None = None
        self._prev_steer: float = 0.0
        self._steps: int = 0
        self._stuck_steps: int = 0
        self._gear: int = 1
        self._last_gear_step: int = -_GEAR_COOLDOWN

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(self, state: SensorState) -> np.ndarray:
        return np.array(
            [
                state.speed / 300.0,
                state.trackPos,
                state.angle / np.pi,
                state.rpm / 10_000.0,
                float(self._gear) / 6.0,
                state.damage / 10_000.0,
                state.track[_TRACK_IDX[0]] / 200.0,
                state.track[_TRACK_IDX[1]] / 200.0,
                state.track[_TRACK_IDX[2]] / 200.0,
            ],
            dtype=np.float32,
        )

    def _update_gear(self, state: SensorState) -> None:
        if (self._steps - self._last_gear_step) < _GEAR_COOLDOWN:
            return
        if state.rpm > _GEAR_UP_RPM and self._gear < 6:
            self._gear += 1
            self._last_gear_step = self._steps
        elif state.rpm < _GEAR_DOWN_RPM and self._gear > 1:
            self._gear -= 1
            self._last_gear_step = self._steps

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        import time

        if self._client is None:
            # Reconnect after timeout or first initialization
            logger.info("Reconnecting to TORCS...")
            time.sleep(1)  # Wait for TORCS to settle after timeout
            self._client = TORCSClient(self._host, self._port)
            try:
                self._client.connect()
            except Exception as e:
                logger.error(f"Failed to reconnect: {e}")
                self._client = None
                raise
        else:
            try:
                self._client.send_restart()
            except Exception as e:
                logger.warning(f"Restart failed ({e}), reconnecting...")
                try:
                    self._client.close()
                except:
                    pass
                self._client = None
                time.sleep(1)
                self._client = TORCSClient(self._host, self._port)
                self._client.connect()

        # Drain protocol control messages until we get a real sensor packet
        result = self._client.receive()
        while result in (RESTART, SHUTDOWN):
            if result == SHUTDOWN:
                self._client.close()
                self._client = None
                raise RuntimeError("TORCS server shut down during env reset")
            result = self._client.receive()

        self._prev_state = result
        self._prev_steer = 0.0
        self._steps = 0
        self._stuck_steps = 0
        self._gear = 1
        self._last_gear_step = -_GEAR_COOLDOWN

        return self._make_obs(result), {}

    def step(self, action: np.ndarray):
        assert self._prev_state is not None, "Call reset() before step()"

        # Reconnect if client was cleared (e.g., after timeout recovery)
        if self._client is None:
            raise RuntimeError("Client is None; reset() must be called first")

        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        self._update_gear(self._prev_state)
        act = Action(steer=steer, accel=accel, brake=brake, gear=self._gear)

        # Send action and receive next state (retry on transient failures)
        try:
            self._client.send(act)
            result = self._client.receive()
        except ConnectionError as e:
            # On timeout, terminate episode and let reset() handle reconnection
            logger.warning(f"Timeout: {e}")
            try:
                self._client.close()
            except:
                pass
            self._client = None
            # Return terminal signal with large negative reward
            obs = self._make_obs(self._prev_state)
            return obs, -100.0, True, False, {"event": "timeout"}

        if result in (RESTART, SHUTDOWN):
            # Server-side forced restart or shutdown — end the episode
            obs = self._make_obs(self._prev_state)
            return obs, -50.0, True, False, {"event": str(result)}

        curr: SensorState = result
        lap_completed = curr.lap > self._prev_state.lap

        reward = compute_reward(
            self._prev_state,
            curr,
            prev_steer=self._prev_steer,
            curr_steer=steer,
            lap_completed=lap_completed,
        )

        # Stuck detection: if the car barely moves for too long, truncate
        if abs(curr.speed) < self._stuck_speed:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        self._prev_state = curr
        self._prev_steer = steer
        self._steps += 1

        terminated = lap_completed
        truncated = (
            self._steps >= self._max_steps
            or self._stuck_steps >= self._stuck_patience
        )

        info = {
            "speed_kmh": curr.speed,
            "trackPos": curr.trackPos,
            "damage": curr.damage,
            "lap": curr.lap,
            "stuck_steps": self._stuck_steps,
        }
        return self._make_obs(curr), reward, terminated, truncated, info

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.send_shutdown()
                self._client.close()
            except Exception:
                pass
            self._client = None
