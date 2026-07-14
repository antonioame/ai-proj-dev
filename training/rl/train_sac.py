"""Entry point di training SAC per la Fase 3 (REINFORCEMENT_LEARNING.md Sezioni 6 e 9).

Di default questo script lancia da sé un TORCS headless (con la working
directory corretta — vedi il commento su TORCS_EXE più sotto) una volta
costruito il modello SAC, così il client si connette prima che scatti il
breve timeout di pre-connessione di TORCS. Passa --no-launch-torcs per
connetterti invece a un server avviato separatamente.
Fa il warm-start dell'attore dal modello BC per le curve (bc_from_olddriver_v1)
a meno che non sia passato --no-warmstart.

Usage:
    conda run -n ai_env python training/rl/train_sac.py \\
        --total-timesteps 200000 --reward-version baseline_v1

Riprendere un run interrotto (es. dopo una caduta della connessione TORCS/SCR):
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

# Corrisponde a hidden_dims=[128, 64] di BCPolicy così le forme dei layer
# dell'attore con warm-start combaciano esattamente con ciò che si aspetta
# load_bc_backbone_into_actor().
NET_ARCH = [128, 64]


class EpisodeCheckpointCallback(BaseCallback):
    """Salva un checkpoint ogni `episodes_per_checkpoint` episodi completati
    (Sezione 8: "Checkpoint every ~50 episodes")."""

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
    parser.add_argument("--reward-version", choices=["baseline_v1", "refined_v2", "safe_progress_v3", "safe_progress_v4"], default="baseline_v1")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--critic-warmup-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                         help="SAC learning rate (actor+critic+ent_coef optimizer), SB3 default 3e-4. "
                              "In every direct-mode run so far the actor degrades as soon as it starts "
                              "updating, regardless of reward/entropy/warmup tuning — a much smaller "
                              "value (e.g. 5e-5) fine-tunes a warm-started actor without destructive "
                              "drift, mirroring what fixed the analogous BC fine-tuning collapse.")
    parser.add_argument("--ent-coef", default=None,
                        help="SAC entropy coefficient. Default: 0.02 for --residual, \"auto\" otherwise. "
                             "Override to a small fixed value (e.g. 0.02) in direct mode to stop SB3's "
                             "auto entropy tuning from drifting a warm-started actor away from BC.")
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
        # Il residual RL mantiene l'esplorazione delicata (un coefficiente di
        # entropia fisso e piccolo) così la correzione resta vicina alla base
        # BC; la variante ad azione diretta auto-regola l'entropia da zero.
        # CRITICO: raccogliere un intero episodio prima di fare gli update del
        # gradiente (train_freq per episodio), invece dell'update del
        # gradiente di default di SB3 dopo ogni singolo step. TORCS `-r`
        # headless gira sul proprio clock e NON aspetta un client lento —
        # continua ad avanzare la simulazione con l'ultima azione. Un update
        # del gradiente per-step (~10-30ms su CPU) è un ritardo sufficiente
        # perché l'auto derivi dalla linea di gara al lancio ad alta velocità
        # e si schianti, il che ha corrotto silenziosamente ogni run RL
        # precedente (gli episodi finivano fuori pista dopo ~300 step a
        # prescindere dalla policy — verificato forzando l'azione a puro BC e
        # schiantandosi comunque con il training per-step).
        # Fare gli update del gradiente tra un episodio e l'altro (auto ferma
        # o in fase di reset) mantiene bassa la latenza per-step così l'auto
        # guida davvero.
        model = WarmStartSAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=NET_ARCH),
            learning_rate=args.learning_rate,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "episode"),
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef if args.ent_coef is not None else (0.02 if args.residual else "auto"),
            critic_warmup_steps=args.critic_warmup_steps,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(Path(__file__).resolve().parent / "tb_logs"),
        )
        warm_started_from = None
        if args.residual:
            # Nessun trasferimento di pesi BC: il driver BC *è* la base. Azzera
            # l'attore così il residuo iniziale è 0 → il training parte guidando
            # come puro BC.
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
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef if args.ent_coef is not None else (0.02 if args.residual else "auto"),
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

    # TORCS viene lanciato e chiuso per-episodio dall'ambiente stesso
    # (TorcsSacEnv, auto_launch_torcs) — vedi quel modulo per il motivo per
    # cui si rilancia invece di usare un restart in-gara.
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
