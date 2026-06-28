"""Test SB3 training loop timing."""
import time
import numpy as np
from stable_baselines3 import PPO
from training.rl.gym_env import TORCSGymEnv

env = TORCSGymEnv(host="localhost", port=3001, max_steps=100)

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=16,  # Small batch for quick testing
    batch_size=8,
    n_epochs=1,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    device="cpu",
)

print("Starting SB3 training (20 steps total)...")
t0 = time.perf_counter()

try:
    model.learn(total_timesteps=20, log_interval=1)
    t1 = time.perf_counter()
    elapsed = (t1 - t0)
    print(f"\n✓ Training completed in {elapsed:.1f}s ({elapsed/20*1000:.1f}ms per step)")
except Exception as e:
    t1 = time.perf_counter()
    elapsed = (t1 - t0)
    print(f"\n✗ Training failed after {elapsed:.1f}s")
    print(f"Error: {e}")

env.close()
