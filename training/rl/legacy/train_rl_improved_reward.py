"""RL training with improved reward function (v1).

This version emphasizes steering and lane-keeping over raw speed.
"""

from __future__ import annotations

import argparse
import logging
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
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="RL training with improved reward (v1)")
    parser.add_argument("--target-steps", type=int, default=20000, help="Target total steps")
    parser.add_argument("--sessions", type=int, default=15, help="Max sessions")
    parser.add_argument("--save-path", default="models/rl_improved_reward_v1")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 RL Training with Improved Reward (v1)")
    logger.info(f"Target: {args.target_steps:,} steps across {args.sessions} sessions")
    logger.info(f"Model: {save_path}")
    logger.info("")
    logger.info("Reward Changes:")
    logger.info("  - PROGRESS_WEIGHT: 1.0 → 0.5 (less emphasis on speed)")
    logger.info("  - DEVIATION_WEIGHT: 1.0 → 5.0 (5× penalty for drift)")
    logger.info("  - OFF_TRACK_PENALTY: 2.0 → 20.0 (10× penalty when off track)")
    logger.info("  - Added progressive penalties: 0.5→5pts, 0.75→10pts, 1.0→20pts")
    logger.info("  - Added LANE_CENTER_BONUS: reward for staying centered")
    logger.info("")

    total_steps = 0
    session_num = 0
    model_path = save_path / "model.zip"

    for session_num in range(1, args.sessions + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"SESSION {session_num} | Progress: {total_steps:,}/{args.target_steps:,} steps")
        logger.info(f"{'='*70}")

        torcs_proc = start_torcs()
        time.sleep(2)

        env = TORCSGymEnv(host="localhost", port=3001)

        # Load or create model
        if model_path.exists():
            logger.info(f"Loading checkpoint: {model_path}")
            model = PPO.load(str(model_path), env=env, device="cpu")
        else:
            logger.info("Creating new PPO model")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=1e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=3,
                gamma=0.99,
                gae_lambda=0.95,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                tensorboard_log=None,
                device="cpu",
            )

        session_steps = 0
        try:
            remaining = args.target_steps - total_steps
            t0 = time.time()
            model.learn(total_timesteps=remaining, reset_num_timesteps=False, log_interval=None)
            elapsed = time.time() - t0
            session_steps = remaining
            logger.info(f"✓ Session {session_num}: {session_steps:,} steps in {elapsed:.1f}s")

        except (ConnectionError, RuntimeError) as e:
            if "TORCS" in str(e) or "10054" in str(e):
                elapsed = time.time() - t0
                session_steps = int(600 * elapsed) if elapsed > 0 else 0
                logger.warning(f"Session {session_num} timeout after {elapsed:.1f}s (~{session_steps} steps)")
            else:
                logger.error(f"Session {session_num} error: {e}")

        finally:
            total_steps += session_steps
            try:
                model.save(str(model_path))
            except:
                pass

            try:
                env.close()
            except:
                pass

            stop_torcs(torcs_proc)
            time.sleep(2)

        logger.info(f"Total progress: {total_steps:,}/{args.target_steps:,} ({100*total_steps//args.target_steps}%)")

        if total_steps >= args.target_steps:
            break

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Training Complete!")
    logger.info(f"Total steps: {total_steps:,}")
    logger.info(f"Sessions: {session_num}")
    logger.info(f"Model: {save_path}/model.zip")
    logger.info(f"{'='*70}")

    # Create final checkpoint
    final_path = save_path / "final.zip"
    if model_path.exists():
        import shutil
        shutil.copy(model_path, final_path)
        logger.info(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
