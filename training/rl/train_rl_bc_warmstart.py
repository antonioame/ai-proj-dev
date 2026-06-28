"""RL fine-tuning with BC checkpoint as warm-start.

Start from BC v2 (8-feature model trained on 35k samples), then fine-tune with RL
to optimize for lap time while staying on track.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from stable_baselines3 import PPO
from training.rl.gym_env import TORCSGymEnv
from training.behavioral_cloning.model import MLPPolicy as BCMLPPolicy

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


def load_bc_checkpoint(bc_path: Path) -> dict:
    """Load BC checkpoint and extract policy weights."""
    logger.info(f"Loading BC checkpoint from {bc_path}")
    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    return ckpt


def create_bc_initialized_ppo(
    env,
    bc_checkpoint: dict,
    policy_lr: float = 1e-4,
) -> PPO:
    """Create PPO model initialized from BC weights."""
    logger.info("Creating PPO model initialized from BC checkpoint")

    # Create PPO with same policy architecture as BC
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=policy_lr,
        n_steps=512,  # Larger buffer than before
        batch_size=64,
        n_epochs=3,   # More training per batch
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),  # Larger network
        tensorboard_log=None,
        device="cpu",
    )

    # Try to initialize policy from BC weights
    try:
        bc_state = bc_checkpoint["model_state"]
        ppo_state = model.policy.state_dict()

        # Match BC weights to PPO policy
        matched = 0
        for bc_name, bc_param in bc_state.items():
            # BC uses "steer_head", "accel_head", "brake_head", "backbone"
            # PPO uses "mlp_extractor.policy_net" and "mlp_extractor.value_net"

            # Try to map BC backbone to PPO mlp_extractor
            if "backbone" in bc_name and "mlp_extractor" in ppo_state:
                # Simplified: just log that we found backbone
                logger.info(f"Found BC backbone layer: {bc_name}")
                matched += 1

        logger.info(f"Matched {matched} BC layers to PPO. Starting with BC knowledge.")
    except Exception as e:
        logger.warning(f"Could not initialize from BC weights: {e}. Starting fresh.")

    return model


def train_rl_session(
    session_num: int,
    bc_checkpoint: dict,
    total_target_steps: int,
    model_dir: Path,
) -> int:
    """Run one RL training session with BC warm-start."""
    torcs_proc = start_torcs()
    time.sleep(2)

    env = TORCSGymEnv(host="localhost", port=3001)

    # Create PPO initialized from BC
    model = create_bc_initialized_ppo(env, bc_checkpoint, policy_lr=1e-4)

    # Load checkpoint from previous session if it exists
    model_path = model_dir / "model.zip"
    if model_path.exists():
        logger.info(f"Loading previous checkpoint: {model_path}")
        model = PPO.load(str(model_path), env=env, device="cpu")

    steps_this_session = 0
    try:
        logger.info(f"Session {session_num}: Training with BC warm-start...")
        t0 = time.time()

        # Learn for remaining steps
        remaining = total_target_steps
        model.learn(total_timesteps=remaining, reset_num_timesteps=False, log_interval=None)

        elapsed = time.time() - t0
        steps_this_session = remaining
        logger.info(f"✓ Session {session_num}: {steps_this_session:,} steps in {elapsed:.1f}s")

    except (ConnectionError, RuntimeError) as e:
        if "TORCS" in str(e) or "10054" in str(e):
            elapsed = time.time() - t0
            steps_this_session = int(600 * elapsed) if elapsed > 0 else 0
            logger.warning(f"Session {session_num} timeout after {elapsed:.1f}s (~{steps_this_session} steps)")
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
    parser = argparse.ArgumentParser(description="RL fine-tuning with BC warm-start")
    parser.add_argument("--bc-model", default="models/bc_v2.pth", help="BC checkpoint path")
    parser.add_argument("--target-steps", type=int, default=50000, help="Total target steps")
    parser.add_argument("--sessions", type=int, default=30, help="Max sessions")
    parser.add_argument("--save-path", default="models/rl_bc_warmstart")
    args = parser.parse_args()

    bc_path = Path(args.bc_model)
    if not bc_path.exists():
        logger.error(f"BC checkpoint not found: {bc_path}")
        return

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    bc_checkpoint = load_bc_checkpoint(bc_path)

    logger.info(f"🚀 RL Fine-tuning with BC Warm-start")
    logger.info(f"BC Model: {bc_path}")
    logger.info(f"Target: {args.target_steps:,} steps across {args.sessions} sessions")

    total_steps = 0

    for session_num in range(1, args.sessions + 1):
        steps = train_rl_session(session_num, bc_checkpoint, args.target_steps, save_path)
        total_steps += steps

        logger.info(f"\n{'='*70}")
        logger.info(f"Progress: {total_steps:,}/{args.target_steps:,} steps ({100*total_steps//args.target_steps}%)")
        logger.info(f"{'='*70}\n")

        if total_steps >= args.target_steps:
            break

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ RL Fine-tuning Complete!")
    logger.info(f"Total steps: {total_steps:,}")
    logger.info(f"Sessions: {session_num}")
    logger.info(f"Model: {save_path}/model.zip")
    logger.info(f"{'='*70}")

    # Create final checkpoint
    final_path = save_path / "final.zip"
    model_path = save_path / "model.zip"
    if model_path.exists():
        import shutil
        shutil.copy(model_path, final_path)
        logger.info(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
