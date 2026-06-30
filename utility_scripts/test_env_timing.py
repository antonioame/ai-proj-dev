"""Test gym environment step timing."""
import time
import numpy as np
from training.rl.gym_env import TORCSGymEnv

env = TORCSGymEnv(host="localhost", port=3001, max_steps=1000)
obs, info = env.reset()

print("Testing 20 env.step() calls...")
times = []

for i in range(20):
    action = np.array([0.1, 0.5, 0.0], dtype=np.float32)  # steer, accel, brake

    t0 = time.perf_counter()
    obs, reward, terminated, truncated, info = env.step(action)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    times.append(elapsed_ms)

    status = "TERM" if terminated else "TRUNC" if truncated else "OK"
    print(f"Step {i:2d}: {elapsed_ms:7.1f} ms [{status}]")

    if terminated or truncated:
        print(f"Episode ended at step {i}, resetting...")
        obs, info = env.reset()

if times:
    print(f"\nAverage: {sum(times)/len(times):.1f} ms")
    print(f"Max: {max(times):.1f} ms")
    print(f"95th percentile: {sorted(times)[int(len(times)*0.95)]:.1f} ms")
    print(f"TORCS timeout: 2850 ms")
    if max(times) > 2850:
        print("⚠️  TIMING EXCEEDED!")
    else:
        print("✓ OK")

env.close()
