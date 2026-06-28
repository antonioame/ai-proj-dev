"""Train a PPO agent on TORCS (Phase 3).

PPO is easier to tune than DDPG but collects experience on-policy, making it
slower to converge. Prefer DDPG for sample efficiency; use PPO if DDPG is unstable.

Usage:
    TORCS_HOST=<windows-ip> conda run -n ai_env python training/rl/train_ppo.py

Usage (resume):
    python training/rl/train_ppo.py --load models/ppo_v1 --steps 500000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.rl.gym_env import TORCSGymEnv


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on TORCS")
    parser.add_argument("--host", default=os.environ.get("TORCS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TORCS_PORT", "3001")))
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--save-path", default="models/ppo_v1", help="Output model path (no extension)")
    parser.add_argument("--load", default=None, help="Resume training from this checkpoint (.zip)")
    parser.add_argument("--device", default=_best_device())
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_path.parent / "checkpoints_ppo"

    print(f"Device: {args.device} | Host: {args.host}:{args.port} | Steps: {args.steps:,}")

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
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            tensorboard_log=str(save_path.parent / "tb_ppo"),
            device=args.device,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=str(ckpt_dir),
        name_prefix="ppo",
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint_cb,
            log_interval=10,
            reset_num_timesteps=args.load is None,
        )
    finally:
        model.save(str(save_path))
        print(f"\nModel saved → {save_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
