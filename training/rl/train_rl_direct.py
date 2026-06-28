"""Direct RL training: custom loop, no SB3 wrappers, timeout-resilient with checkpoints.

Bypasses SB3's Monitor/VecEnv overhead by implementing a minimal training loop.
Checkpoints after each batch for recovery from timeouts.

Usage:
    conda run -n ai_env python training/rl/train_rl_direct.py --steps 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.rl.gym_env import TORCSGymEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_with_checkpoints(
    env: TORCSGymEnv,
    model: PPO,
    total_steps: int,
    checkpoint_dir: Path,
    batch_size: int = 512,
):
    """Train with checkpoint recovery on timeout."""
    steps_done = 0
    episode_count = 0
    batch_count = 0

    while steps_done < total_steps:
        batch_count += 1
        batch_steps = min(batch_size, total_steps - steps_done)

        logger.info(f"\n=== Batch {batch_count} ({batch_steps} steps) ===")

        try:
            # Collect rollouts
            logger.info("Collecting rollouts...")
            t0 = time.time()
            model.learn(
                total_timesteps=batch_steps,
                reset_num_timesteps=False,
                log_interval=None,  # No logging overhead
            )
            elapsed = time.time() - t0
            logger.info(f"Rollouts complete: {batch_steps} steps in {elapsed:.1f}s ({batch_steps/elapsed:.1f} steps/s)")

            steps_done += batch_steps

            # Checkpoint after successful batch
            checkpoint_path = checkpoint_dir / f"rl_step_{steps_done}.zip"
            model.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Log progress
            meta_file = checkpoint_dir / "progress.json"
            with open(meta_file, "w") as f:
                json.dump({"steps_done": steps_done, "batch_count": batch_count}, f)

        except (ConnectionError, RuntimeError) as e:
            if "TORCS" in str(e) or "10054" in str(e) or "reset()" in str(e):
                logger.warning(f"TORCS error (batch {batch_count}): {e}")
                # Close and reset env, wait for TORCS to recover
                try:
                    env.close()
                except:
                    pass
                # Re-create environment and wait longer
                logger.info("Waiting 5s for TORCS recovery...")
                time.sleep(5)
                try:
                    env = TORCSGymEnv(host=env._host, port=env._port)
                    # Try a quick reset to verify connection
                    obs, info = env.reset()
                    logger.info("Reconnection successful, resuming training")
                except Exception as e2:
                    logger.error(f"Failed to reconnect: {e2}, skipping to completion")
                    break
                continue
            else:
                raise

    logger.info(f"\n✓ Training complete! Total steps: {steps_done}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct RL training with checkpoint recovery")
    parser.add_argument("--host", default=os.environ.get("TORCS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TORCS_PORT", "3001")))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512, help="Steps per checkpoint")
    parser.add_argument("--save-path", default="models/rl_direct_v1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_path.parent / "rl_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Direct RL Training (custom loop, timeout-resilient)")
    logger.info(f"Steps: {args.steps:,} | Batch: {args.batch_size} | Device: {args.device}")
    logger.info(f"Checkpoints: {checkpoint_dir}")

    # Raw env, no wrapping
    env = TORCSGymEnv(host=args.host, port=args.port)

    # Minimal PPO for CPU
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-3,
        n_steps=128,
        batch_size=32,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[64], vf=[64])),
        tensorboard_log=None,
        device=args.device,
    )

    try:
        train_with_checkpoints(env, model, args.steps, checkpoint_dir, args.batch_size)
    finally:
        model.save(str(save_path / "final"))
        logger.info(f"Final model saved: {save_path}/final.zip")
        env.close()


if __name__ == "__main__":
    main()
