"""Residual-RL environment: SAC learns a small correction on top of the full
working BC driver, instead of trying to replace it.

Why this exists
---------------
The first RL attempt (train_sac.py without --residual) warm-started SAC from a
single raw BC sub-network (bc_from_olddriver_v1) — but the driver that actually
completes the Corkscrew lap is the *whole* `_DRIVER.driver.BCDriver` pipeline:
a blend of two networks, post-hoc STEER/ACCEL/BRAKE gains, RPM gear management
and a startup phase. Warm-starting one sub-network's weights is NOT the same as
starting from the working driver, and SAC's entropy exploration then eroded
even that into a car that stalls (0 laps completed).

Residual RL fixes both problems:
  final_action = BCDriver.step(state)  +  RESIDUAL_SCALE * rl_residual
with `rl_residual` in [-1, 1]^3 and the SAC actor zeroed at init (see
zero_residual_actor), so at the start of training the agent drives *exactly*
like the 121.978s BC driver and completes laps immediately. RL then learns
small, bounded, state-dependent nudges to go faster. Because the residual is
bounded it can't catastrophically stall or leave the track the way the
from-scratch policy did — the BC base keeps the car alive.

This reuses TorcsSacEnv wholesale (per-episode TORCS relaunch, reward,
termination, observation) and only changes how the control command is built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _DRIVER.driver import BCDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.torcs_gym_env import TorcsSacEnv

# Max physical magnitude of the RL correction on each of steer/accel/brake, and
# the L2 penalty that keeps the trained residual close to the BC base.
#
# These two together make the shipped residual driver both work (completes the
# lap, 0% off-track) and be genuinely RL (a trained SAC policy adjusts the base
# every step). The shipped checkpoint (drivers/rl/models/sac_corkscrew_residual)
# was trained at these values and evaluates deterministically at 127.07s, 0%
# off-track, 0 damage (vs BC 121.978s / 0%).
#   * Scale bounds how far RL can pull the car off the BC line. A constant
#     worst-case "never brake" attack overshoots one corner (~91% around) at
#     every scale 0.02-0.05, but that's a dumb attack that brakes nowhere; a
#     policy trained on clean laps (with the -200 off-track penalty) learns to
#     brake into corners, so 0.03 is safe for a trained policy.
#   * RESIDUAL_L2_COEF penalises ||residual||^2 each step, so a correction must
#     buy back its own cost in driving reward — the policy defaults to near-pure
#     BC and only nudges where it clearly helps. With this + clean per-episode
#     training (see train_sac SAC config comment on why per-step training
#     corrupted every earlier run) the learned residual keeps the car exactly on
#     BC's line (0% off-track).
# Note: this does NOT beat BC on lap time (~4% slower) — the goal was a working,
# genuinely-RL driver that safely completes the lap. Beating BC would need a
# lap-time reward, not a scale tweak.
RESIDUAL_SCALE = 0.03
RESIDUAL_L2_COEF = 5.0


class ResidualTorcsSacEnv(TorcsSacEnv):
    """TorcsSacEnv where the action is a bounded residual on the BC driver."""

    def __init__(self, *args, residual_scale: float = RESIDUAL_SCALE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._res_scale = residual_scale
        self._bc = BCDriver()
        # SAC actor works in a normalised [-1, 1]^3 residual space; the physical
        # scale is applied here. Standard SAC target-entropy math wants [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Reset the BC base driver so its startup phase and gear state restart
        # with the episode; it owns the launch (see _run_startup override).
        self._bc.on_restart()
        return super().reset(seed=seed, options=options)

    def _run_startup(self, state: SensorState) -> SensorState:
        # No env-driven startup burst: the BC base driver handles the standing
        # start itself over its own first BCDriver.STARTUP_STEPS steps.
        return state

    def step(self, residual):
        self._ensure_started()  # deferred launch+connect on first step (see base env)
        d = np.asarray(residual, dtype=np.float32)
        base = self._bc.step(self._last_state)
        cmd = Action(
            steer=base.steer + float(d[0]) * self._res_scale,
            accel=base.accel + float(d[1]) * self._res_scale,
            brake=base.brake + float(d[2]) * self._res_scale,
            gear=base.gear,
        )
        obs, reward, terminated, truncated, info = self._send_and_observe(cmd)
        # Penalise deviating from the BC base (see RESIDUAL_L2_COEF).
        reward -= RESIDUAL_L2_COEF * float(np.sum(d * d))
        return obs, reward, terminated, truncated, info


def zero_residual_actor(model) -> None:
    """Zero the SAC actor's output layer so the initial residual mean is 0 —
    i.e. training starts driving exactly like the BC base — and start with a
    *very* small exploration std.

    The exploration std matters a lot here: it's added on top of a BC driver
    that already threads Corkscrew's tight corners, so even modest steering
    noise (an earlier try at std≈0.22 → ~±0.06 physical steer jitter/step)
    accumulates and knocks the car off-track ~285 steps in, before it ever
    completes a lap — so the agent never sees the reward for finishing. log_std
    = -3.0 (std≈0.05 → ~±0.0075 physical steer) keeps exploration gentle enough
    that the car stays on track and episodes run full laps, giving SAC a real
    lap-completion signal to learn from. The deterministic policy still has the
    full ±RESIDUAL_SCALE authority — only the training-time noise is small.
    """
    actor = model.policy.actor
    with torch.no_grad():
        actor.mu.weight.zero_()
        actor.mu.bias.zero_()
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-3.0)
