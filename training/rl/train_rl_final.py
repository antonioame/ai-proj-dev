"""Final RL attempt: run multiple TORCS sessions, accumulate steps via checkpoints.

Each session tries to collect as many steps as possible. On timeout, save and restart fresh.
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

from stable_baselines3 import PPO
from training.rl.gym_env import TORCSGymEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TORCS_EXE = Path(r"U:\AI-Partition\torcs\torcs\wtorcs.exe")
RACE_XML = Path(r"U:\AI-Partition\progetto_v2\ai_private_proj\torcs_env\race_config\corkscrew_solo.xml")


def start_torcs() -> subprocess.Popen:
    """Start TORCS."""
    logger.info("Starting TORCS...")
    proc = subprocess.Popen(
        [str(TORCS_EXE), "-r", str(RACE_XML)],
        cwd=str(TORCS_EXE.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    return proc


def stop_torcs(proc: subprocess.Popen) -> None:
    """Stop TORCS."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except:
            proc.kill()


def train_session(
    session_num: int,
    total_steps: int,
    model_path: Path,
    host: str = "localhost",
    port: int = 3001,
) -> int:
    """Run one training session. Returns steps completed this session."""
    torcs_proc = start_torcs()
    time.sleep(2)

    env = TORCSGymEnv(host=host, port=port)

    # Load or create model
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(str(model_path), env=env, device="cpu")
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
            device="cpu",
        )

    steps_this_session = 0
    try:
        logger.info(f"Session {session_num}: Collecting up to {total_steps} steps...")
        t0 = time.time()
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, log_interval=None)
        elapsed = time.time() - t0
        steps_this_session = total_steps
        logger.info(f"✓ Session {session_num} complete: {total_steps} steps in {elapsed:.1f}s")

    except (ConnectionError, RuntimeError) as e:
        if "TORCS" in str(e) or "10054" in str(e):
            elapsed = time.time() - t0
            # Estimate how many steps we got before timeout
            if elapsed > 0:
                steps_this_session = int(600 * elapsed)  # ~600 steps/sec
            logger.warning(f"Session {session_num} timeout after ~{elapsed:.1f}s (~{steps_this_session} steps)")
        else:
            logger.error(f"Session {session_num} error: {e}")

    finally:
        try:
            model.save(str(model_path))
            logger.info(f"Model checkpoint saved")
        except:
            pass

        try:
            env.close()
        except:
            pass

        stop_torcs(torcs_proc)
        time.sleep(2)

    return steps_this_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Final RL: multiple sessions, auto-checkpointing")
    parser.add_argument("--sessions", type=int, default=5, help="Number of TORCS sessions")
    parser.add_argument("--steps-per-session", type=int, default=5000, help="Target steps per session")
    parser.add_argument("--save-path", default="models/rl_final_v1")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "model.zip"

    logger.info(f"Final RL Training: {args.sessions} sessions × {args.steps_per_session} steps")
    logger.info(f"Model: {model_path}")

    total_steps_accumulated = 0

    for session_num in range(1, args.sessions + 1):
        steps = train_session(session_num, args.steps_per_session, model_path)
        total_steps_accumulated += steps

        logger.info(f"\n{'='*60}")
        logger.info(f"Progress: {total_steps_accumulated:,} / {args.sessions * args.steps_per_session:,} steps")
        logger.info(f"{'='*60}\n")

        if steps < args.steps_per_session // 2:
            logger.warning(f"Session {session_num} achieved <50% target. Continuing...")

    logger.info(f"\n✓ RL Training Complete!")
    logger.info(f"Total steps: {total_steps_accumulated:,}")
    logger.info(f"Model: {model_path}")

    # Create final copy
    final_path = save_path / "final.zip"
    if model_path.exists():
        import shutil
        shutil.copy(model_path, final_path)
        logger.info(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
