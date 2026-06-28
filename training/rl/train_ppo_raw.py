"""Train PPO directly with raw env (no DummyVecEnv wrapper overhead).

Usage:
    conda run -n ai_env python training/rl/train_ppo_raw.py --steps 200000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.rl.gym_env import TORCSGymEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on TORCS (raw env, no Monitor)")
    parser.add_argument("--host", default=os.environ.get("TORCS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TORCS_PORT", "3001")))
    parser.add_argument("--steps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--save-path", default="models/ppo_v1", help="Output model path (no extension)")
    parser.add_argument("--load", default=None, help="Resume training from this checkpoint (.zip)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device} | Host: {args.host}:{args.port} | Steps: {args.steps:,}")
    print(f"Using raw TORCSGymEnv (no DummyVecEnv/Monitor overhead)")

    # Raw env without wrapping
    env = TORCSGymEnv(host=args.host, port=args.port)

    if args.load:
        model = PPO.load(args.load, env=env, device=args.device)
        print(f"Resumed from {args.load}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            tensorboard_log=str(save_path.parent / "tb_ppo"),
            device=args.device,
        )

    try:
        model.learn(
            total_timesteps=args.steps,
            log_interval=10,
            reset_num_timesteps=args.load is None,
        )
    finally:
        model.save(str(save_path))
        print(f"\nModel saved → {save_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
