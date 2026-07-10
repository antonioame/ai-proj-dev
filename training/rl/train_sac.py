"""SAC training entry point for Phase 3 (REINFORCEMENT_LEARNING.md Sections 6 & 9).

By default this auto-launches a headless TORCS itself (with the correct working
directory — see the TORCS_EXE comment below) once the SAC model is built, so the
client connects before TORCS's short pre-connection timeout fires. Pass
--no-launch-torcs to instead connect to a server you started separately.
Warm-starts the actor from the BC corner model (bc_from_olddriver_v1) unless
--no-warmstart is given.

Usage:
    conda run -n ai_env python training/rl/train_sac.py \\
        --total-timesteps 200000 --reward-version baseline_v1

Resume an interrupted run (e.g. after a TORCS/SCR connection drop):
    conda run -n ai_env python training/rl/train_sac.py \\
        --resume training/rl/checkpoints/<run_id>/checkpoint_ep150.zip \\
        --total-timesteps 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3.common.callbacks import BaseCallback

from training.rl.sac_warmstart import DEFAULT_BC_CHECKPOINT, WarmStartSAC, load_bc_backbone_into_actor
from training.rl.torcs_gym_env import TorcsSacEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
RUN_LOG_DIR = Path(__file__).resolve().parent / "run_logs"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "drivers" / "rl" / "models" / "sac_corkscrew_v1.zip"

# Matches BCPolicy's hidden_dims=[128, 64] so the warm-started actor's layer
# shapes line up exactly with what load_bc_backbone_into_actor() expects.
NET_ARCH = [128, 64]


class EpisodeCheckpointCallback(BaseCallback):
    """Saves a checkpoint every `episodes_per_checkpoint` completed episodes
    (Section 8: "Checkpoint every ~50 episodes")."""

    def __init__(self, save_dir: Path, episodes_per_checkpoint: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.episodes_per_checkpoint = episodes_per_checkpoint
        self._episodes_done = 0

    def _on_step(self) -> bool:
        for done in self.locals.get("dones", []):
            if done:
                self._episodes_done += 1
                if self._episodes_done % self.episodes_per_checkpoint == 0:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    ckpt = self.save_dir / f"checkpoint_ep{self._episodes_done}.zip"
                    self.model.save(str(ckpt))
                    logger.info("Checkpoint saved: %s (episode %d)", ckpt, self._episodes_done)
        return True


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 3 SAC driver")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--reward-version", choices=["baseline_v1", "refined_v2"], default="baseline_v1")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--critic-warmup-steps", type=int, default=3000)
    parser.add_argument("--gradient-steps", type=int, default=512,
                        help="Gradient updates done between episodes (train_freq is per-episode; "
                             "see the SAC config comment on why per-step training corrupts driving).")
    parser.add_argument("--episodes-per-checkpoint", type=int, default=50)
    parser.add_argument("--bc-checkpoint", default=str(DEFAULT_BC_CHECKPOINT))
    parser.add_argument("--no-warmstart", action="store_true", help="Skip BC warm-start, train SAC from scratch")
    parser.add_argument(
        "--residual", action="store_true",
        help="Residual RL: learn a bounded correction on top of the full BC driver "
             "(ResidualTorcsSacEnv) instead of replacing it. Best fix for the "
             "from-scratch policy stalling — starts driving exactly like BC.",
    )
    parser.add_argument("--resume", default=None, help="Path to an existing SAC .zip checkpoint to resume from")
    parser.add_argument("--save-path", default=str(DEFAULT_SAVE_PATH))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-launch-torcs", action="store_true",
        help="Don't auto-launch TORCS; connect to a server already running on the SCR port.",
    )
    args = parser.parse_args()

    if args.residual:
        from training.rl.residual_env import ResidualTorcsSacEnv, zero_residual_actor
        env = ResidualTorcsSacEnv(
            host=args.host,
            port=args.port,
            reward_version=args.reward_version,
            auto_launch_torcs=not args.no_launch_torcs,
        )
    else:
        env = TorcsSacEnv(
            host=args.host,
            port=args.port,
            reward_version=args.reward_version,
            auto_launch_torcs=not args.no_launch_torcs,
        )

    if args.resume:
        logger.info("Resuming SAC model from %s", args.resume)
        model = WarmStartSAC.load(args.resume, env=env, critic_warmup_steps=args.critic_warmup_steps)
        reset_num_timesteps = False
        warm_started_from = None
    else:
        # Residual RL keeps exploration gentle (a fixed small entropy coef) so
        # the correction stays close to the BC base; the direct-action variant
        # auto-tunes entropy from scratch.
        # CRITICAL: collect a whole episode before doing gradient updates
        # (train_freq per episode), instead of SB3's default gradient update
        # after every single step. TORCS `-r` headless runs on its own clock and
        # does NOT wait for a late client — it keeps advancing the sim with the
        # last action. A per-step gradient update (~10-30ms on CPU) is enough lag
        # for the car to drift off the racing line at the high-speed launch and
        # crash, which silently corrupted every earlier RL run (episodes ended
        # off-track ~300 steps in regardless of the policy — verified by forcing
        # the action to pure BC and still crashing under per-step training).
        # Doing the gradient updates between episodes (car stationary/resetting)
        # keeps the per-step latency low so the car actually drives.
        model = WarmStartSAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=NET_ARCH),
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "episode"),
            gradient_steps=args.gradient_steps,
            ent_coef=0.02 if args.residual else "auto",
            critic_warmup_steps=args.critic_warmup_steps,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(Path(__file__).resolve().parent / "tb_logs"),
        )
        warm_started_from = None
        if args.residual:
            # No BC weight transfer: the BC driver *is* the base. Zero the actor
            # so the initial residual is 0 → training starts driving as pure BC.
            zero_residual_actor(model)
            warm_started_from = "residual(BC base + zeroed actor)"
        elif not args.no_warmstart:
            load_bc_backbone_into_actor(model, Path(args.bc_checkpoint))
            warm_started_from = str(args.bc_checkpoint)
        reset_num_timesteps = True

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "residual" if args.residual else "direct"
    run_config = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "algorithm": "SAC",
        "mode": mode_tag,
        "reward_version": args.reward_version,
        "total_timesteps": args.total_timesteps,
        "gradient_steps_per_episode": args.gradient_steps,
        "critic_warmup_steps": args.critic_warmup_steps,
        "warm_started_from": warm_started_from,
        "resumed_from": args.resume,
        "net_arch": NET_ARCH,
        "seed": args.seed,
    }
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_log_path = RUN_LOG_DIR / f"{run_id}_{mode_tag}_{args.reward_version}.json"
    run_log_path.write_text(json.dumps(run_config, indent=2))
    logger.info("Run config logged: %s", run_log_path)

    checkpoint_cb = EpisodeCheckpointCallback(
        save_dir=CHECKPOINT_DIR / run_id, episodes_per_checkpoint=args.episodes_per_checkpoint
    )

    # TORCS is launched and torn down per-episode by the env itself
    # (TorcsSacEnv, auto_launch_torcs) — see that module for why we relaunch
    # instead of using an in-race restart.
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=f"sac_{mode_tag}_{args.reward_version}",
        )
    finally:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        logger.info("Final model saved: %s", save_path)
        env.close()


if __name__ == "__main__":
    main()
