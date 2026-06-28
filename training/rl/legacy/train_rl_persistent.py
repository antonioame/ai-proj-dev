"""Persistent RL trainer: auto-restart TORCS on timeout, resume from checkpoint.

Runs training in small batches, automatically restarts TORCS when it times out,
and resumes from the last saved checkpoint. Will keep training until target steps.

Usage:
    conda run -n ai_env python training/rl/train_rl_persistent.py --steps 50000 --target-steps 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.rl.train_rl_direct import train_with_checkpoints
from training.rl.gym_env import TORCSGymEnv
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TORCS_EXE = Path(r"U:\AI-Partition\torcs\torcs\wtorcs.exe")
RACE_XML = Path(r"U:\AI-Partition\progetto_v2\ai_private_proj\torcs_env\race_config\corkscrew_solo.xml")


def start_torcs() -> subprocess.Popen:
    """Start TORCS headless server."""
    logger.info("Starting TORCS...")
    proc = subprocess.Popen(
        [str(TORCS_EXE), "-r", str(RACE_XML)],
        cwd=str(TORCS_EXE.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)  # Wait for startup
    return proc


def stop_torcs(proc: subprocess.Popen) -> None:
    """Stop TORCS process."""
    if proc and proc.poll() is None:
        logger.info("Stopping TORCS...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def load_checkpoint(checkpoint_dir: Path) -> int:
    """Load training progress from checkpoint."""
    progress_file = checkpoint_dir / "progress.json"
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                data = json.load(f)
                return data.get("steps_done", 0)
        except:
            pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent RL training with auto-restart")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--steps", type=int, default=10_000, help="Steps per batch")
    parser.add_argument("--target-steps", type=int, default=50_000, help="Total target steps")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-path", default="models/rl_persistent_v1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_path.parent / "rl_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Persistent RL Training")
    logger.info(f"Target: {args.target_steps:,} steps | Batch size: {args.batch_size} | Device: {args.device}")
    logger.info(f"Checkpoints: {checkpoint_dir}")

    # Load progress
    steps_done = load_checkpoint(checkpoint_dir)
    logger.info(f"Resuming from checkpoint: {steps_done:,} / {args.target_steps:,} steps")

    torcs_proc = None
    attempt = 0
    max_attempts = 20  # Max TORCS restarts

    try:
        while steps_done < args.target_steps and attempt < max_attempts:
            attempt += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Attempt {attempt}: Starting batch ({steps_done:,}/{args.target_steps:,} steps)")
            logger.info(f"{'='*60}")

            # Start TORCS
            torcs_proc = start_torcs()
            time.sleep(2)

            # Create environment and model
            env = TORCSGymEnv(host=args.host, port=args.port)

            # Load or create model
            model_file = checkpoint_dir.parent / "rl_persistent_v1.zip"
            if model_file.exists():
                logger.info(f"Loading model from {model_file}")
                model = PPO.load(str(model_file), env=env, device=args.device)
            else:
                logger.info("Creating new PPO model")
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

            # Train this batch
            batch_steps = min(args.steps, args.target_steps - steps_done)
            try:
                logger.info(f"Collecting {batch_steps} steps...")
                t0 = time.time()
                model.learn(total_timesteps=batch_steps, reset_num_timesteps=False, log_interval=None)
                elapsed = time.time() - t0

                steps_done += batch_steps
                logger.info(f"✓ Batch complete: {batch_steps} steps in {elapsed:.1f}s ({batch_steps/elapsed:.1f} steps/s)")

                # Save checkpoint
                model.save(str(model_file))
                progress_file = checkpoint_dir / "progress.json"
                with open(progress_file, "w") as f:
                    json.dump({"steps_done": steps_done, "attempt": attempt}, f)
                logger.info(f"Checkpoint saved: {steps_done:,} steps")

            except (ConnectionError, RuntimeError) as e:
                if "TORCS" in str(e) or "10054" in str(e):
                    logger.warning(f"TORCS timeout/error: {e}")
                    logger.info("Will restart TORCS and resume...")
                else:
                    raise

            finally:
                # Clean up
                try:
                    env.close()
                except:
                    pass
                stop_torcs(torcs_proc)
                time.sleep(2)

        # Done
        if steps_done >= args.target_steps:
            logger.info(f"\n✓ Training complete! {steps_done:,}/{args.target_steps:,} steps")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Final model: {model_file}")
        else:
            logger.warning(f"Training stopped at {steps_done:,}/{args.target_steps:,} steps after {attempt} attempts")

    finally:
        if torcs_proc:
            stop_torcs(torcs_proc)


if __name__ == "__main__":
    main()
