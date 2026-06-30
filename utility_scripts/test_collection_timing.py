"""Test rollout collection timing without training."""
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from training.rl.gym_env import TORCSGymEnv

# Wrap env like SB3 does
def make_env():
    return Monitor(TORCSGymEnv(host="localhost", port=3001, max_steps=100))

env = DummyVecEnv([make_env])
obs = env.reset()

print("Collecting 20 steps through DummyVecEnv...")
times = []

for i in range(20):
    action = env.action_space.sample()  # Random action

    t0 = time.perf_counter()
    obs, rewards, dones, infos = env.step([action])
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    times.append(elapsed_ms)
    print(f"Step {i:2d}: {elapsed_ms:7.1f} ms")

if times:
    print(f"\nAverage: {sum(times)/len(times):.1f} ms")
    print(f"Max: {max(times):.1f} ms")
    if max(times) > 2850:
        print("⚠️  EXCEEDED TORCS TIMEOUT")

env.close()
