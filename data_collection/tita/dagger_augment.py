"""Raccoglie esempi di RECUPERO per il candidato bc_tita_v2, usando bc
(_DRIVER.driver.BCDriver, il driver di produzione, sicuro e gia' validato)
come rete di sicurezza.

Idea: bc_tita_v2 e' stato addestrato solo su giri "perfetti" di Tita, quindi
non ha mai visto un esempio di correzione da una traiettoria imprecisa. Se
mentre guida esce troppo dalla linea (|trackPos| oltre SAFETY_TRIGGER), il
controllo passa a bc finche' non rientra stabilmente in pista (sotto
SAFETY_RESUME per RESUME_HYSTERESIS tick consecutivi); durante quel tratto
si registrano stato sensori + azione di bc (la correzione "giusta") in un
CSV con lo stesso schema del dataset BC.

Non e' DAgger in senso stretto (non si interroga l'oracolo ad ogni step),
ma un approccio pragmatico "safety-net": il risultato e' un dataset di
esempi di recupero da unire ai giri puliti di Tita per il prossimo
candidato (bc_tita_v3).

Non tocca _DRIVER/, drivers/, scripts/, training/: legge bc solo in lettura
(BCDriver di _DRIVER.driver, mai modificato) e scrive in
data_collection/tita/dagger_recovery/.

NOTA sulle unita' delle label (rilevata in audit 2026-07-17): le azioni
registrate qui sono quelle di bc DOPO i gain post-hoc e il clamp
(safety_action = output finale di BCDriver.step), mentre i CSV di tita
convertiti contengono azioni raw del bot. All'inferenza il driver candidato
riapplica i propri gain (ACCEL x1.40, BRAKE x0.80) a TUTTO l'output del
modello -> per i campioni di recovery il gain risulta di fatto applicato due
volte. Empiricamente tollerato (bc_tita_v20, addestrato anche su questi
campioni, completa il giro pulito a 111.986s), ma da normalizzare (dividere
le label bc per i gain, o registrare l'output pre-gain) PRIMA di eventuali
futuri round di raccolta.

Prerequisito: TORCS avviato con torcs_env/race_config/corkscrew_solo.xml
(client SCR, stesso schema di scripts/run/run_agent.py).

Usage:
    python data_collection/tita/dagger_augment.py --laps 5
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _DRIVER.driver import BCDriver
from data_collection.tita.driver_candidate import BCTitaCandidateDriver
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAFETY_TRIGGER = 0.55    # |trackPos| oltre cui subentra bc (abbassato da 0.70:
                          # a volte anche bc non riusciva piu' a recuperare da
                          # +1.4, probabilmente gia' incastrata contro un muro)
SAFETY_RESUME = 0.30     # |trackPos| sotto cui si puo' tornare al candidato
RESUME_HYSTERESIS = 30   # tick consecutivi sotto SAFETY_RESUME prima di ridare il controllo
STATUS_EVERY = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "dagger_recovery"


def run(
    laps: int = 5,
    host: Optional[str] = None,
    port: Optional[int] = None,
    checkpoint_name: str = "bc_tita_v2",
) -> dict:
    candidate_dir = Path(__file__).resolve().parent / "candidate_models"
    candidate = BCTitaCandidateDriver(
        checkpoint_path=candidate_dir / f"{checkpoint_name}.pth",
        stats_path=candidate_dir / f"{checkpoint_name}.npz",
    )
    safety = BCDriver()

    mode = "candidate"
    resume_counter = 0
    recovery_rows: list[dict] = []
    lap_times: list[float] = []
    lap_count = 0
    # Giro (state.lap) all'ultima registrazione: rileva un nuovo giro anche se
    # due giri consecutivi hanno tempo identico (simulazione deterministica) —
    # stesso doppio criterio di scripts/eval/evaluate_common.py.
    lap_at_last_record = 0
    total_steps = 0
    safety_steps = 0
    off_track_steps = 0
    max_speed = 0.0

    with TORCSClient(host=host, port=port) as client:
        logger.info("Starting DAgger-style recovery collection for %d lap(s).", laps)

        try:
            while True:
                result = client.receive()

                if result == SHUTDOWN:
                    logger.info("Server shutdown.")
                    break
                if result == RESTART:
                    logger.info("Server restart signal.")
                    candidate.on_restart()
                    safety.on_restart()
                    mode = "candidate"
                    resume_counter = 0
                    lap_count = 0
                    lap_times = []
                    lap_at_last_record = 0
                    continue

                state = result

                # Facciamo avanzare lo stato interno di ENTRAMBI i driver ad
                # ogni tick (step_count, current_gear), cosi' quando si passa
                # dall'uno all'altro nessuno dei due riparte da uno stato
                # "congelato".
                candidate_action = candidate.step(state)
                safety_action = safety.step(state)

                abs_pos = abs(state.trackPos)

                if mode == "candidate" and abs_pos > SAFETY_TRIGGER:
                    mode = "safety"
                    resume_counter = 0
                    logger.info("-> SAFETY NET on (|trackPos|=%.2f)", abs_pos)
                elif mode == "safety":
                    if abs_pos < SAFETY_RESUME:
                        resume_counter += 1
                        if resume_counter >= RESUME_HYSTERESIS:
                            mode = "candidate"
                            logger.info("-> back to CANDIDATE (recovered)")
                    else:
                        resume_counter = 0

                action = safety_action if mode == "safety" else candidate_action
                client.send(action)

                total_steps += 1
                max_speed = max(max_speed, state.speed)
                if abs_pos > 1.0:
                    off_track_steps += 1

                if mode == "safety":
                    safety_steps += 1
                    recovery_rows.append({
                        "angle": state.angle,
                        "speed": state.speed,
                        "speedY": state.speedY,
                        "speedZ": state.speedZ,
                        "trackPos": state.trackPos,
                        **{f"track_{i}": state.track[i] for i in range(len(state.track))},
                        "rpm": state.rpm,
                        "gear": state.gear,
                        "steer": safety_action.steer,
                        "accel": safety_action.accel,
                        "brake": safety_action.brake,
                    })

                if total_steps % STATUS_EVERY == 0:
                    logger.info(
                        "lap %d | %6.0f m | %5.1f km/h | pos %+.2f | mode=%s",
                        state.lap, state.distFromStart, state.speed, state.trackPos, mode,
                    )

                if state.lastLapTime > 0 and (
                    not lap_times
                    or state.lastLapTime != lap_times[-1]
                    or state.lap > lap_at_last_record
                ):
                    lap_times.append(state.lastLapTime)
                    lap_count += 1
                    lap_at_last_record = state.lap
                    logger.info("Lap %d completed in %.3f s (safety-net active for %d/%d ticks so far)",
                                lap_count, state.lastLapTime, safety_steps, total_steps)
                    if lap_count >= laps:
                        logger.info("Target laps reached — releasing control to TORCS.")
                        break
        except ConnectionError as exc:
            logger.warning("Connessione interrotta a meta' corsa (%s) — salvo comunque gli esempi raccolti finora.", exc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"recovery_{timestamp}.csv"

    if recovery_rows:
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(recovery_rows[0].keys()))
            writer.writeheader()
            writer.writerows(recovery_rows)
        logger.info("Recovery examples: %s (%d rows)", out_path, len(recovery_rows))
    else:
        logger.info("No recovery examples collected (candidate never left the safe zone).")

    results = {
        "laps_completed": lap_count,
        "lap_times": lap_times,
        "total_steps": total_steps,
        "safety_steps": safety_steps,
        "safety_pct": round(100.0 * safety_steps / max(total_steps, 1), 2),
        "off_track_pct": round(100.0 * off_track_steps / max(total_steps, 1), 2),
        "max_speed_kmh": round(max_speed, 2),
        "recovery_rows": len(recovery_rows),
        "recovery_csv": str(out_path) if recovery_rows else None,
    }
    logger.info("Summary: %s", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bc-safety-net recovery examples for the tita candidate")
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="bc_tita_v2")
    args = parser.parse_args()

    run(laps=args.laps, host=args.host, port=args.port, checkpoint_name=args.checkpoint_name)


if __name__ == "__main__":
    main()
