"""Cross-Entropy Method (CEM): ottimizzazione black-box dei pesi della rete,
alternativa strutturale al SAC diretto (TD-learning).

Perché: 9 run SAC diretti (reward/entropia/learning rate/rumore diversi)
mostrano tutti lo stesso pattern, la policy resta stabile finché l'attore è
congelato e degrada non appena partono gli update del gradiente TD, mai un
miglioramento netto su BC (121,978s). Non è un problema di iperparametri: è
il TD-learning stesso (bootstrap su un critic il cui errore si propaga
nell'attore) a essere strutturalmente instabile in questo ambiente.

CEM evita il problema alla radice: nessun critic, nessuna backpropagation su
una value function. Perturba i pesi della policy, valuta ogni variante con
un giro reale su TORCS (fitness = tempo sul giro vero) e tiene solo le
varianti migliori: nel caso peggiore una generazione non migliora e si
riprova dalla media corrente, non può collassare per un gradiente distruttivo.

Rete: HybridCemPolicy (drivers/cem/driver.py), due sotto-reti rettilineo/curva
fuse su track[9] come _DRIVER/driver.py.BCDriver, non un modello singolo: una
CemPolicy a rete singola si è bloccata a 142,87s (21s sotto la vera BC) per
mancanza di questa specializzazione.

Non tocca i file di Fase 1/2 né i checkpoint SAC esistenti: nuovi checkpoint
in drivers/rl/models/cem_*.

Usage:
    conda run -n ai_env python training/rl/train_cem.py \\
        --generations 15 --population 8 --elite 3 --sigma 0.02
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from drivers.cem.driver import HybridCemPolicy
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.features import build_feature_vector
from training.rl.torcs_gym_env import TorcsSacEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRAIGHT_CHECKPOINT = PROJECT_ROOT / "_DRIVER" / "models" / "bc_from_attempt1_v1.pth"
CORNER_CHECKPOINT = PROJECT_ROOT / "_DRIVER" / "models" / "bc_from_olddriver_v1.pth"
NORM_STATS_PATH = CORNER_CHECKPOINT.with_suffix(".npz")
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_cem"
RUN_LOG_DIR = Path(__file__).resolve().parent / "run_logs"

# Guadagni post-hoc identici a BCDriver/RLDriver/torcs_gym_env.py.
_STEER_GAIN = 1.8
_ACCEL_GAIN = 1.40
_BRAKE_GAIN = 0.80
_STARTUP_STEPS = 80
_GEAR_UP_RPM = 12000.0
_GEAR_DOWN_RPM = 6000.0

# Un'uscita di pista precoce non deve "vincere" su un giro incompleto ma
# lungo: penalità fissa pesante + piccolo bonus di progresso per la distanza
# comunque coperta, così CEM ha comunque un gradiente di fitness utile tra
# due candidati che falliscono entrambi (non solo tra chi completa e chi no).
_INCOMPLETE_PENALTY = 300.0
_PROGRESS_BONUS_PER_METER = 0.05
_DAMAGE_PENALTY_PER_POINT = 0.5


def _startup_gear(speed: float) -> int:
    if speed < 15.0:
        return 1
    if speed < 45.0:
        return 2
    return 3


def _load_hybrid_from_bc() -> HybridCemPolicy:
    policy = HybridCemPolicy()
    straight_state = torch.load(STRAIGHT_CHECKPOINT, map_location="cpu")
    corner_state = torch.load(CORNER_CHECKPOINT, map_location="cpu")
    with torch.no_grad():
        for sub, state in [(policy.straight, straight_state), (policy.corner, corner_state)]:
            sub.backbone[0].weight.copy_(state["backbone.0.weight"])
            sub.backbone[0].bias.copy_(state["backbone.0.bias"])
            sub.backbone[2].weight.copy_(state["backbone.2.weight"])
            sub.backbone[2].bias.copy_(state["backbone.2.bias"])
            sub.head_steer.weight.copy_(state["head_steer.weight"])
            sub.head_steer.bias.copy_(state["head_steer.bias"])
            sub.head_accel.weight.copy_(state["head_accel.weight"])
            sub.head_accel.bias.copy_(state["head_accel.bias"])
            sub.head_brake.weight.copy_(state["head_brake.weight"])
            sub.head_brake.bias.copy_(state["head_brake.bias"])
    return policy


def get_flat_params(policy: HybridCemPolicy) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in policy.parameters()])


def set_flat_params(policy: HybridCemPolicy, flat: np.ndarray) -> None:
    offset = 0
    for p in policy.parameters():
        numel = p.numel()
        chunk = flat[offset : offset + numel].reshape(p.shape)
        with torch.no_grad():
            p.copy_(torch.from_numpy(chunk).float())
        offset += numel


@dataclass
class EpisodeResult:
    fitness: float
    lap_time: float | None
    dist_reached: float
    off_track_pct: float
    damage: float
    completed: bool


def run_episode(env: TorcsSacEnv, policy: HybridCemPolicy) -> EpisodeResult:
    """Fa girare un episodio con *policy* (pesi correnti) e ne calcola il
    fitness dal vero esito della gara, non dal reward RL shapato."""
    env.reset()

    step_count = 0
    current_gear = 1
    off_track_steps = 0
    total_steps = 0
    last_info: dict = {}
    terminated = truncated = False

    while not (terminated or truncated):
        env._ensure_started()  # noqa: SLF001
        state: SensorState = env._last_state  # noqa: SLF001
        step_count += 1

        if step_count <= _STARTUP_STEPS:
            # L'env ha già eseguito 80 step di burst in _run_startup, quindi qui
            # l'auto riceve circa 160 step di pieno gas contro gli 80 di
            # evaluate.py --driver cem. Lasciato invariato apposta: i checkpoint
            # cem_v1..v5 sono stati selezionati sotto questo profilo, cambiarlo
            # renderebbe i fitness storici non confrontabili.
            gear = _startup_gear(state.speed)
            current_gear = gear
            cmd = Action(steer=0.0, accel=1.0, brake=0.0, gear=gear)
        else:
            raw = build_feature_vector(state).astype(np.float32)
            front_dist = state.track[9] if len(state.track) > 9 else 100.0
            with torch.no_grad():
                out = policy(torch.from_numpy(raw), front_dist).numpy()
            steer, accel, brake = out.tolist()

            if state.rpm > _GEAR_UP_RPM and current_gear < 6:
                current_gear += 1
            elif state.rpm < _GEAR_DOWN_RPM and current_gear > 1:
                current_gear -= 1

            cmd = Action(
                steer=steer * _STEER_GAIN,
                accel=accel * _ACCEL_GAIN,
                brake=brake * _BRAKE_GAIN,
                gear=current_gear,
            )

        _, _, terminated, truncated, info = env._send_and_observe(cmd.clamp())  # noqa: SLF001
        total_steps += 1
        if abs(info.get("trackPos", 0.0)) > 1.0:
            off_track_steps += 1
        last_info = info

    dist_reached = last_info.get("distRaced", 0.0)
    damage = last_info.get("damage", 0.0)
    off_track_pct = (off_track_steps / max(total_steps, 1)) * 100.0
    completed = last_info.get("termination_reason") == "lap_completed"
    lap_time = last_info.get("lastLapTime") if completed else None

    if completed and lap_time and lap_time > 0:
        fitness = -lap_time
    else:
        fitness = -_INCOMPLETE_PENALTY + _PROGRESS_BONUS_PER_METER * dist_reached

    fitness -= _DAMAGE_PENALTY_PER_POINT * damage

    return EpisodeResult(
        fitness=fitness, lap_time=lap_time, dist_reached=dist_reached,
        off_track_pct=off_track_pct, damage=damage, completed=completed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM black-box optimization of the hybrid BC-architecture policy")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.02, help="Initial per-parameter perturbation std")
    parser.add_argument("--sigma-decay", type=float, default=0.95, help="Multiply sigma by this each generation")
    parser.add_argument("--reward-version", default="safe_progress_v4", help="Only used for episode termination logic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=str(PROJECT_ROOT / "drivers" / "rl" / "models" / "cem_v1.pth"))
    parser.add_argument("--resume-from", default=None,
                         help="Path to an existing HybridCemPolicy checkpoint to use as the starting "
                              "theta_mean instead of the exact BC weights, to continue a previous search.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    base_policy = HybridCemPolicy()
    if args.resume_from:
        base_policy.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
        logger.info("Resumed starting weights from %s", args.resume_from)
    else:
        base_policy = _load_hybrid_from_bc()
        logger.info(
            "Loaded exact hybrid BC weights (straight=%s, corner=%s), no adapter",
            STRAIGHT_CHECKPOINT.name, CORNER_CHECKPOINT.name,
        )

    theta_mean = get_flat_params(base_policy)
    n_params = theta_mean.shape[0]
    sigma = np.full(n_params, args.sigma, dtype=np.float64)
    logger.info("Parameter vector size: %d", n_params)

    env = TorcsSacEnv(reward_version=args.reward_version, auto_launch_torcs=True)
    candidate = HybridCemPolicy()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_fitness = -np.inf
    best_theta = theta_mean.copy()
    history = []

    try:
        for gen in range(1, args.generations + 1):
            # Il candidato 0 di ogni generazione è sempre theta_mean stesso
            # (nessuna perturbazione), così possiamo tracciare il progresso
            # "pulito" della media a fianco delle perturbazioni esplorative.
            thetas = [theta_mean.copy()] + [
                theta_mean + rng.normal(0.0, sigma) for _ in range(args.population - 1)
            ]

            results: list[tuple[float, EpisodeResult, np.ndarray]] = []
            for i, theta in enumerate(thetas):
                set_flat_params(candidate, theta)
                res = run_episode(env, candidate)
                logger.info(
                    "gen %d/%d cand %d/%d: fitness=%.2f lap=%s dist=%.0f off_track=%.1f%% damage=%.1f",
                    gen, args.generations, i + 1, args.population,
                    res.fitness, f"{res.lap_time:.3f}s" if res.lap_time else "n/a",
                    res.dist_reached, res.off_track_pct, res.damage,
                )

                # Un candidato che batte il record va riverificato con un
                # secondo giro indipendente: pesi che guidano un giro perfetto
                # falliscono a volte al reload (fino al 70% fuori pista), un
                # singolo giro può essere "fortunato". Si usa il peggiore dei
                # due esiti (per il best e per la selezione elite), così un
                # candidato fragile non scavalca uno robusto per un colpo di fortuna.
                if res.fitness > best_fitness:
                    set_flat_params(candidate, theta)
                    res_verify = run_episode(env, candidate)
                    logger.info(
                        "  verification run: fitness=%.2f lap=%s off_track=%.1f%% damage=%.1f",
                        res_verify.fitness,
                        f"{res_verify.lap_time:.3f}s" if res_verify.lap_time else "n/a",
                        res_verify.off_track_pct, res_verify.damage,
                    )
                    if res_verify.fitness < res.fitness:
                        res = res_verify

                results.append((res.fitness, res, theta))

                if res.fitness > best_fitness:
                    best_fitness = res.fitness
                    best_theta = theta.copy()
                    set_flat_params(candidate, best_theta)
                    torch.save(candidate.state_dict(), CHECKPOINT_DIR / f"{run_id}_best_so_far.pth")
                    logger.info("New best (verified): fitness=%.2f (checkpoint saved)", best_fitness)

            results.sort(key=lambda r: r[0], reverse=True)
            elite = results[: args.elite]
            elite_thetas = np.stack([e[2] for e in elite])
            theta_mean = elite_thetas.mean(axis=0)
            sigma = sigma * args.sigma_decay + 1e-4  # non azzerare mai del tutto l'esplorazione

            gen_summary = {
                "generation": gen,
                "best_fitness_this_gen": results[0][0],
                "elite_mean_fitness": float(np.mean([e[0] for e in elite])),
                "best_fitness_overall": best_fitness,
                "sigma_mean": float(sigma.mean()),
            }
            history.append(gen_summary)
            logger.info("Generation %d summary: %s", gen, gen_summary)
    finally:
        env.close()
        save_path = Path(args.save_path)
        if save_path.exists():
            # Non sovrascrivere mai un checkpoint promosso/consegnato già presente
            # (es. il default cem_v1.pth), salva a fianco con il run_id.
            save_path = save_path.parent / f"{save_path.stem}_{run_id}{save_path.suffix}"
            logger.warning("Existing checkpoint preserved, saved to %s instead", save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        set_flat_params(candidate, best_theta)
        torch.save(candidate.state_dict(), save_path)
        logger.info("Best CEM policy saved: %s (fitness=%.2f)", save_path, best_fitness)

        run_log_path = RUN_LOG_DIR / f"{run_id}_cem.json"
        run_log_path.write_text(json.dumps({
            "run_id": run_id,
            "algorithm": "CEM",
            "generations": args.generations,
            "population": args.population,
            "elite": args.elite,
            "initial_sigma": args.sigma,
            "sigma_decay": args.sigma_decay,
            "reward_version": args.reward_version,
            "best_fitness": best_fitness,
            "save_path": str(save_path),
            "history": history,
        }, indent=2))
        logger.info("Run log: %s", run_log_path)


if __name__ == "__main__":
    main()
