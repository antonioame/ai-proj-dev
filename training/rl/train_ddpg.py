"""Train a DDPG agent on TORCS (Phase 3).

Usage (Mac/Linux):
    TORCS_HOST=<windows-ip> conda run -n ai_env python training/rl/train_ddpg.py

Usage (resume from checkpoint):
    python training/rl/train_ddpg.py --load models/ddpg_v1 --steps 200000

The script auto-detects the best device (MPS on Apple Silicon, CUDA on Nvidia, CPU otherwise).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.rl.gym_env import TORCSGymEnv


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDPG agent on TORCS")
    parser.add_argument("--host", default=os.environ.get("TORCS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TORCS_PORT", "3001")))
    parser.add_argument("--steps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--save-path", default="models/ddpg_v1", help="Output model path (no extension)")
    parser.add_argument("--load", default=None, help="Resume training from this checkpoint (.zip)")
    parser.add_argument("--device", default=_best_device())
    parser.add_argument("--noise-sigma", type=float, default=0.1, help="Exploration noise σ")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_path.parent / "checkpoints_ddpg"

    print(f"Device: {args.device} | Host: {args.host}:{args.port} | Steps: {args.steps:,}")

    env = TORCSGymEnv(host=args.host, port=args.port)
    n_actions = env.action_space.shape[0]
    noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise_sigma * np.ones(n_actions),
    )

    if args.load:
        model = DDPG.load(args.load, env=env, device=args.device)
        print(f"Resumed from {args.load}")
    else:
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=2_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            action_noise=noise,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            tensorboard_log=str(save_path.parent / "tb_ddpg"),
            device=args.device,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(ckpt_dir),
        name_prefix="ddpg",
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint_cb,
            log_interval=100,
            reset_num_timesteps=args.load is None,
        )
    finally:
        model.save(str(save_path))
        print(f"\nModel saved → {save_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
