"""RL fine-tuning with BC checkpoint as warm-start.

Start from BC v2 (8-feature model, z-score normalised), then fine-tune with PPO.

Key fixes vs previous version:
  1. Model is built BEFORE TORCS starts — eliminates pre-connection timeout.
  2. BC backbone weights are properly mapped to PPO policy_net layers.
  3. Real step count tracked via SB3 callback (no more fake elapsed-time estimates).
  4. Per-session step limit avoids over-running a single TORCS session.
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
from stable_baselines3.common.callbacks import BaseCallback
from training.rl.gym_env import TORCSGymEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TORCS_EXE = Path(r"U:\AI-Partition\torcs\torcs\wtorcs.exe")
RACE_XML = Path(r"U:\AI-Partition\progetto_v2\ai_private_proj\torcs_env\race_config\corkscrew_solo.xml")

# Steps to collect per TORCS session before saving a checkpoint.
# Keep below ~1 500 to avoid hitting the per-session TORCS timeout.
STEPS_PER_SESSION = 1000


class StepCounter(BaseCallback):
    """Callback that counts real environment steps."""

    def __init__(self) -> None:
        super().__init__()
        self.real_steps: int = 0

    def _on_step(self) -> bool:
        self.real_steps += 1
        return True  # continue training


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
        except Exception:
            proc.kill()


def load_bc_checkpoint(bc_path: Path) -> dict:
    logger.info("Loading BC checkpoint from %s", bc_path)
    return torch.load(str(bc_path), map_location="cpu", weights_only=False)


def _init_ppo_from_bc(model: PPO, bc_state: dict) -> int:
    """Copy BC backbone weights into PPO's policy_net.

    BC backbone layout (indices into nn.Sequential):
        0 Linear(8→256)   ← copy → PPO policy_net[0] Linear(8→256)
        1 LayerNorm(256)
        2 ReLU
        3 Linear(256→256) ← copy → PPO policy_net[2] Linear(256→256)
        4 LayerNorm(256)
        5 ReLU
        6 Linear(256→128) (no matching layer in PPO — skip)

    Returns the number of parameter tensors copied.
    """
    ppo_state = model.policy.state_dict()
    patch: dict[str, torch.Tensor] = {}

    mapping = {
        "backbone.0.weight": "mlp_extractor.policy_net.0.weight",
        "backbone.0.bias":   "mlp_extractor.policy_net.0.bias",
        "backbone.3.weight": "mlp_extractor.policy_net.2.weight",
        "backbone.3.bias":   "mlp_extractor.policy_net.2.bias",
    }

    copied = 0
    for bc_key, ppo_key in mapping.items():
        if bc_key in bc_state and ppo_key in ppo_state:
            bc_param = bc_state[bc_key]
            ppo_param = ppo_state[ppo_key]
            if bc_param.shape == ppo_param.shape:
                patch[ppo_key] = bc_param.clone()
                copied += 1
            else:
                logger.warning(
                    "Shape mismatch %s: BC %s vs PPO %s — skipped",
                    bc_key, bc_param.shape, ppo_param.shape,
                )

    if patch:
        ppo_state.update(patch)
        model.policy.load_state_dict(ppo_state)
        logger.info("Copied %d parameter tensors from BC backbone to PPO policy_net", copied)
    else:
        logger.warning("No BC weights copied — starting from random initialisation")

    return copied


def build_model(env: TORCSGymEnv, bc_checkpoint: dict, model_path: Path) -> PPO:
    """Build PPO from scratch or load existing checkpoint, then patch in BC weights."""
    if model_path.exists():
        logger.info("Loading PPO checkpoint: %s", model_path)
        model = PPO.load(str(model_path), env=env, device="cpu")
        logger.info("Checkpoint loaded — skipping BC weight init")
        return model

    logger.info("Creating fresh PPO model")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=STEPS_PER_SESSION,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=None,
        device="cpu",
    )
    _init_ppo_from_bc(model, bc_checkpoint["model_state"])
    return model


def train_session(
    session_num: int,
    model: PPO,
    model_path: Path,
) -> tuple[int, subprocess.Popen | None]:
    """Run one TORCS session and return (real_steps, torcs_proc).

    TORCS is started AFTER the model is ready so the pre-connection
    timeout cannot fire before the env connects.
    """
    torcs_proc = start_torcs()
    counter = StepCounter()
    t0 = time.time()

    try:
        logger.info("Session %d: collecting %d steps...", session_num, STEPS_PER_SESSION)
        model.learn(
            total_timesteps=STEPS_PER_SESSION,
            callback=counter,
            reset_num_timesteps=False,
            log_interval=None,
        )
        elapsed = time.time() - t0
        logger.info(
            "Session %d done: %d real steps in %.1fs",
            session_num, counter.real_steps, elapsed,
        )

    except (ConnectionError, RuntimeError, OSError) as exc:
        elapsed = time.time() - t0
        logger.warning(
            "Session %d ended early after %.1fs (%d steps): %s",
            session_num, elapsed, counter.real_steps, exc,
        )

    finally:
        try:
            model.save(str(model_path))
        except Exception as exc:
            logger.error("Failed to save checkpoint: %s", exc)

        stop_torcs(torcs_proc)
        time.sleep(1)

    return counter.real_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="RL fine-tuning with BC warm-start")
    parser.add_argument("--bc-model", default="models/bc_v2.pth")
    parser.add_argument("--target-steps", type=int, default=100_000)
    parser.add_argument("--sessions", type=int, default=120)
    parser.add_argument("--save-path", default="models/rl_bc_warmstart_v3_fixed")
    args = parser.parse_args()

    bc_path = Path(args.bc_model)
    if not bc_path.exists():
        logger.error("BC checkpoint not found: %s", bc_path)
        sys.exit(1)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "model.zip"

    bc_checkpoint = load_bc_checkpoint(bc_path)

    # Build a dummy env just for PPO construction (no TORCS needed yet).
    # A real env connects lazily in reset(), so this is safe.
    dummy_env = TORCSGymEnv(host="localhost", port=3001)
    model = build_model(dummy_env, bc_checkpoint, model_path)
    # The dummy env is discarded; each session creates its own env via model.set_env().

    logger.info("Target: %d steps across up to %d sessions", args.target_steps, args.sessions)

    total_steps = 0
    for session_num in range(1, args.sessions + 1):
        remaining = args.target_steps - total_steps
        if remaining <= 0:
            break

        # Recreate env each session so TORCS reconnects cleanly.
        env = TORCSGymEnv(host="localhost", port=3001)
        model.set_env(env)

        real_steps = train_session(session_num, model, model_path)
        total_steps += real_steps

        pct = 100 * total_steps // args.target_steps
        logger.info(
            "Progress: %d/%d steps (%d%%)", total_steps, args.target_steps, pct
        )

        try:
            env.close()
        except Exception:
            pass

    # Save final checkpoint
    final_path = save_path / "final.zip"
    import shutil
    if model_path.exists():
        shutil.copy(model_path, final_path)

    logger.info(
        "Done. Total real steps: %d. Model: %s", total_steps, final_path
    )


if __name__ == "__main__":
    main()
