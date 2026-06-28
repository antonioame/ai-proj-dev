"""Minimal PPO training: smallest network, no logging, CPU-optimized.

Disables Monitor wrapper and all logging to minimize overhead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.rl.gym_env import TORCSGymEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal PPO (CPU-optimized)")
    parser.add_argument("--host", default=os.environ.get("TORCS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TORCS_PORT", "3001")))
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--save-path", default="models/ppo_minimal")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Minimal PPO: tiny network, no logging, CPU-only")
    print(f"Steps: {args.steps:,} | Device: {args.device}")

    # Raw env, no Monitor/Vec wrapping
    env = TORCSGymEnv(host=args.host, port=args.port)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,  # No logging
        learning_rate=1e-3,  # Faster convergence
        n_steps=64,  # Tiny batch
        batch_size=16,
        n_epochs=1,  # One epoch per update
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[32], vf=[32])),  # Tiny network: 32 hidden units
        tensorboard_log=None,  # No tensorboard
        device=args.device,
    )

    print("Starting training (no output to avoid overhead)...")
    try:
        model.learn(total_timesteps=args.steps)
        print(f"\n✓ Training complete!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
    finally:
        model.save(str(save_path))
        print(f"Model saved → {save_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
