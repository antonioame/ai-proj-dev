"""Gymnasium wrapper around the existing TORCS SCR client.

Purely additive: imports torcs_env.client.TORCSClient as-is and orchestrates
it from the outside. No changes to torcs_env/client.py, sensors.py or
actions.py (REINFORCEMENT_LEARNING.md Section 1 / Section 6.3).

Action space is steer/accel/brake only — gear stays automatic (RPM-based,
same thresholds _DRIVER/driver.py uses), matching Section 3's requirement
that the RL action space stay identical to what Phase 1/2 already use so the
BC-warm-started actor's output layout lines up with the environment.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torcs_env.actions import Action
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient
from torcs_env.sensors import SensorState
from training.rl.features import FEATURE_DIM, build_feature_vector
from training.rl.reward import REWARD_VERSIONS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Automatic-gear thresholds, mirrored from _DRIVER/driver.py.BCDriver so the
# RL policy and the eventual RLDriver see the same gear behaviour.
_GEAR_UP_RPM = 12000.0
_GEAR_DOWN_RPM = 6000.0

# Startup grace period: feed full-throttle/zero-steer for this many steps
# before handing control to the policy. Mirrors BCDriver.STARTUP_STEPS —
# keeps the warm-started actor from ever seeing the OOD (speed≈0, gear=0)
# launch state it never saw during BC training.
_STARTUP_STEPS = 80

# Episode termination thresholds.
_OFF_TRACK_STEPS_LIMIT = 25  # ~0.5s at 50 steps/s — "aggressive termination" per Section 8
_STANDING_STILL_KMH = 5.0
_STANDING_STILL_STEPS_LIMIT = 150  # ~3s
_MAX_EPISODE_STEPS = 20000  # a Corkscrew lap is ~13.6k steps at 50/s (see memory: torcs_setup)
_EPISODE_START_RETRIES = 4

# TORCS process management. Each episode gets a FRESH TORCS process rather than
# an in-race meta=1 restart: empirically that in-race restart is unreliable on
# this setup (the connection half-recovers then drops a few steps into the next
# episode — the same instability the earlier, removed Phase 3 attempt hit).
# Relaunching sidesteps it entirely. Two launch requirements, both learned the
# hard way and both load-bearing:
#   1. CWD must be the TORCS install dir, or TORCS can't resolve the car's
#      category files ("Bad Car category for driver scr_server 1") and never
#      binds the SCR port.
#   2. Connect only AFTER a short grace once the port is bound — TORCS binds the
#      port a moment before its sim loop can actually stream sensors, and
#      connecting into that gap deadlocks.
TORCS_EXE = Path(os.environ.get("TORCS_EXE", r"U:\AI-Partition\torcs\torcs\wtorcs.exe"))
RACE_XML = PROJECT_ROOT / "torcs_env" / "race_config" / "corkscrew_solo.xml"
_TORCS_PORT_TIMEOUT = 30.0
_TORCS_STARTUP_GRACE = 4.0

DEFAULT_NORM_STATS_PATH = (
    PROJECT_ROOT / "_DRIVER" / "models" / "bc_from_olddriver_v1.npz"
)


def _port_is_bound(port: int) -> bool:
    """True if something already holds *port* on UDP (i.e. TORCS is up)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


class TorcsSacEnv(gym.Env):
    """One episode = one lap attempt (ends on lap completion, sustained
    off-track excursion, or sustained standing-still)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        reward_version: str = "baseline_v1",
        norm_stats_path: Path = DEFAULT_NORM_STATS_PATH,
        max_episode_steps: int = _MAX_EPISODE_STEPS,
        auto_launch_torcs: bool = True,
    ) -> None:
        super().__init__()
        if reward_version not in REWARD_VERSIONS:
            raise ValueError(f"Unknown reward_version {reward_version!r}; choices: {list(REWARD_VERSIONS)}")
        self.reward_version = REWARD_VERSIONS[reward_version]
        self.max_episode_steps = max_episode_steps
        self._auto_launch = auto_launch_torcs
        self._port = int(port or os.environ.get("TORCS_PORT", "3001"))

        stats = np.load(norm_stats_path)
        self._obs_mean = stats["mean"].astype(np.float32)
        self._obs_std = stats["std"].astype(np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._client = TORCSClient(host=host, port=port)
        self._torcs_proc: Optional[subprocess.Popen] = None
        self._connected = False
        self._current_gear = 1
        self._step_in_episode = 0
        self._off_track_run = 0
        self._standing_run = 0
        self._prev_lap_time = 0.0
        self._last_state: Optional[SensorState] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_gear = 1
        self._step_in_episode = 0
        self._off_track_run = 0
        self._standing_run = 0

        state = self._start_episode()

        self._prev_lap_time = state.lastLapTime
        self._last_state = state
        obs = self._normalize(build_feature_vector(state))
        return obs, {"distRaced": state.distRaced}

    def step(self, action):
        steer, accel, brake = (float(a) for a in np.asarray(action, dtype=np.float32))
        self._update_gear(self._last_state.rpm)
        cmd = Action(steer=steer, accel=accel, brake=brake, gear=self._current_gear)
        return self._send_and_observe(cmd)

    def _send_and_observe(self, cmd: Action):
        """Send one control command, await the next sensor packet, and turn it
        into the Gym (obs, reward, terminated, truncated, info) tuple. Shared
        by the direct-action step() and the residual env's step()."""
        self._client.send(cmd.clamp())

        try:
            state = self._await_fresh_state()
        except ConnectionError as exc:
            # SCR connection dropped mid-episode. The (obs, action, next_obs)
            # transition is no longer valid, so end the episode via truncation
            # rather than feeding a corrupted transition into the replay buffer.
            # Mark disconnected so the next reset() relaunches TORCS cleanly.
            logger.warning("Connection dropped mid-episode: %s", exc)
            self._connected = False
            obs = self._normalize(build_feature_vector(self._last_state))
            info = {"termination_reason": "connection_lost"}
            return obs, 0.0, False, True, info

        self._step_in_episode += 1

        reward, terminated, reason = self._reward_and_termination(self._last_state, state)
        truncated = (not terminated) and self._step_in_episode >= self.max_episode_steps

        self._last_state = state
        obs = self._normalize(build_feature_vector(state))
        info = {
            "distRaced": state.distRaced,
            "trackPos": state.trackPos,
            "speed": state.speed,
            "termination_reason": reason,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            self._client.close()
            self._connected = False
        self._kill_torcs()

    # ------------------------------------------------------------------
    # TORCS process management
    # ------------------------------------------------------------------

    def _kill_torcs(self) -> None:
        proc = self._torcs_proc
        self._torcs_proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _launch_torcs(self) -> None:
        """Start a fresh headless TORCS and block until it's ready to drive.

        See the module-level TORCS_EXE comment for why cwd and the post-bind
        grace are both mandatory.
        """
        if not TORCS_EXE.exists():
            raise FileNotFoundError(f"TORCS executable not found: {TORCS_EXE}")
        self._kill_torcs()
        proc = subprocess.Popen(
            [str(TORCS_EXE), "-r", str(RACE_XML)],
            cwd=str(TORCS_EXE.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._torcs_proc = proc
        deadline = time.monotonic() + _TORCS_PORT_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise ConnectionError(f"TORCS exited during startup (code {proc.returncode}).")
            if _port_is_bound(self._port):
                time.sleep(_TORCS_STARTUP_GRACE)
                return
            time.sleep(0.2)
        self._kill_torcs()
        raise ConnectionError(f"TORCS did not open the SCR port within {_TORCS_PORT_TIMEOUT}s.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _await_fresh_state(self) -> SensorState:
        while True:
            result = self._client.receive()
            if result == RESTART:
                continue
            if result == SHUTDOWN:
                raise ConnectionError("TORCS server sent shutdown while the RL env was running.")
            return result

    def _start_episode(self, retries: int = _EPISODE_START_RETRIES) -> SensorState:
        """Bring the car to a fresh, driveable standing start.

        Each episode gets its own TORCS process (see the TORCS_EXE comment on
        why we relaunch instead of using an in-race meta=1 restart). The whole
        sequence — (re)launch, connect, run the startup throttle grace period —
        is retried as a unit, since TORCS launch itself can occasionally crash
        or the SCR handshake can drop on this setup; a clean relaunch recovers
        where an in-race restart could not.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                if self._connected:
                    self._client.close()
                    self._connected = False
                if self._auto_launch:
                    self._launch_torcs()

                self._client.connect()
                self._connected = True
                state = self._await_fresh_state()
                return self._run_startup(state)
            except (ConnectionError, OSError) as exc:
                last_exc = exc
                logger.warning(
                    "Episode start failed (attempt %d/%d): %s", attempt, retries, exc,
                )
                try:
                    self._client.close()
                except Exception:
                    pass
                self._connected = False
        raise ConnectionError(f"Could not start an episode after {retries} attempts.") from last_exc

    def _run_startup(self, state: SensorState) -> SensorState:
        """Feed full-throttle/zero-steer for _STARTUP_STEPS to get the car off
        the standing start before the policy takes over (see _STARTUP_STEPS).
        Overridable: the residual env skips this and lets its BC base driver
        own the launch instead.
        """
        for _ in range(_STARTUP_STEPS):
            gear = self._startup_gear(state.speed)
            self._current_gear = gear
            self._client.send(Action(steer=0.0, accel=1.0, brake=0.0, gear=gear).clamp())
            state = self._await_fresh_state()
        return state

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self._obs_mean) / self._obs_std).astype(np.float32)

    def _startup_gear(self, speed: float) -> int:
        if speed < 15.0:
            return 1
        if speed < 45.0:
            return 2
        return 3

    def _update_gear(self, rpm: float) -> None:
        if rpm > _GEAR_UP_RPM and self._current_gear < 6:
            self._current_gear += 1
        elif rpm < _GEAR_DOWN_RPM and self._current_gear > 1:
            self._current_gear -= 1

    def _reward_and_termination(self, prev_state: SensorState, state: SensorState):
        rv = self.reward_version
        reward = rv.step_fn(prev_state, state)

        off_track = abs(state.trackPos) > 1.0
        self._off_track_run = self._off_track_run + 1 if off_track else 0

        standing = state.speed < _STANDING_STILL_KMH
        self._standing_run = self._standing_run + 1 if standing else 0

        lap_completed = state.lastLapTime > 0 and state.lastLapTime != self._prev_lap_time

        if self._off_track_run >= _OFF_TRACK_STEPS_LIMIT:
            return reward + rv.off_track_penalty, True, "off_track"
        if self._standing_run >= _STANDING_STILL_STEPS_LIMIT:
            return reward + rv.off_track_penalty * 0.25, True, "standing_still"
        if lap_completed:
            return reward + rv.lap_bonus, True, "lap_completed"
        return reward, False, None
