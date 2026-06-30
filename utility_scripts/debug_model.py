#!/usr/bin/env python3
"""Debug: compare model outputs (BC vs RL v3_fixed)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from torcs_env.sensors import SensorState
from drivers.bc.driver import BCDriver
from drivers.rl.driver import RLDriver

# Fake state: car on track, going straight
state = SensorState(
    angle=0.0,
    curLapTime=0.0,
    damage=0.0,
    distFromStart=100.0,
    distRaced=100.0,
    fuel=100.0,
    gear=1,
    lastLapTime=0.0,
    lap=0,
    rpm=3000.0,
    speed=80.0,
    track=np.array([50.0] * 19, dtype=np.float32),
    trackPos=0.0,
    wheelSpinVel=np.array([0.0] * 4, dtype=np.float32),
)

print("[*] Testing model outputs\n")

# Test BC model
print("BC Model (working baseline):")
try:
    bc_driver = BCDriver()
    import time
    time.sleep(2)  # Wait for model to load
    action = bc_driver.step(state)
    print(f"  Action: steer={action.steer:.3f}, accel={action.accel:.3f}, brake={action.brake:.3f}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test RL v3_fixed
print("RL v3_fixed Model:")
try:
    rl_driver = RLDriver(model_path="models/rl_bc_warmstart_v3_fixed/final", algo="ppo")
    import time
    time.sleep(2)  # Wait for model to load
    action = rl_driver.step(state)
    print(f"  Action: steer={action.steer:.3f}, accel={action.accel:.3f}, brake={action.brake:.3f}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test observation space
print("Observation space check:")
from training.rl.gym_env import _OBS_MEAN, _OBS_STD, _TRACK_IDX, OBS_DIM
print(f"  OBS_DIM: {OBS_DIM} (expected: 8)")
print(f"  Track indices: {_TRACK_IDX} (expected: (6, 12, 18))")
print(f"  Mean: {_OBS_MEAN}")
print(f"  Std:  {_OBS_STD}")

# Test obs construction
from training.rl.gym_env import TORCSGymEnv
env = TORCSGymEnv()
obs = env._make_obs(state)
print(f"\n  Observation from gym_env: {obs}")
print(f"  Shape: {obs.shape} (expected: (8,))")
print(f"  Any NaN/Inf? {np.any(np.isnan(obs)) or np.any(np.isinf(obs))}")
