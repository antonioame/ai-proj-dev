"""Loop di valutazione condiviso da tutti gli scripts/evaluate*.py.

receive/step/send, rilevamento tempo giro, percentuale fuori pista, e salvataggio
del JSON di risultati — logica identica prima duplicata in evaluate.py,
evaluate_cem.py, evaluate_bc_dagger.py ed evaluate_rl.py, cambiava solo quale
classe driver veniva istanziata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from torcs_env.client import RESTART, SHUTDOWN, TORCSClient

logger = logging.getLogger(__name__)


def run_eval_loop(
    driver,
    driver_name: str,
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: Path | None = None,
    max_steps: int | None = None,
) -> dict:
    """Esegue *driver* contro un server TORCS per *laps* giri e salva un JSON
    di risultati strutturato. Se *max_steps* è impostato, aborta e registra
    "aborted_no_lap" quando viene superato senza completare i giri richiesti.
    """
    lap_times: list[float] = []
    speed_samples: list[float] = []
    off_track_steps = 0
    total_steps = 0
    max_damage = 0.0
    lap_count = 0
    # Giro (state.lap) al momento dell'ultima registrazione in lap_times: usato
    # come guard anti-doppio-conteggio e per rilevare un nuovo giro anche
    # quando lastLapTime è identico al giro precedente (simulazione
    # deterministica a parità di codice: due giri consecutivi possono avere
    # lo stesso tempo al millesimo, e il solo confronto sul tempo non lo vedrebbe).
    lap_at_last_record = 0
    aborted_no_lap = False

    with TORCSClient(host=host, port=port) as client:
        logger.info("Evaluating '%s' for %d lap(s).", driver_name, laps)

        while True:
            result = client.receive()

            if result == SHUTDOWN:
                break
            if result == RESTART:
                driver.on_restart()
                # La gara è ripartita da zero: i giri registrati prima del
                # restart non valgono più (stesso reset di run_agent_common).
                lap_count = 0
                lap_times = []
                lap_at_last_record = 0
                continue

            state = result
            action = driver.step(state)
            client.send(action)

            total_steps += 1
            speed_samples.append(state.speed)
            max_damage = max(max_damage, state.damage)

            if abs(state.trackPos) > 1.0:
                off_track_steps += 1

            # Rileva un nuovo giro completato. Non basta confrontare lastLapTime:
            # la simulazione è deterministica a parità di codice, quindi due giri
            # consecutivi possono avere lo stesso tempo al millesimo — in quel
            # caso serve il contatore state.lap (derivato dai reset di distRaced)
            # per accorgersi comunque che è iniziato un nuovo giro.
            if state.lastLapTime > 0 and (
                not lap_times
                or state.lastLapTime != lap_times[-1]
                or state.lap > lap_at_last_record
            ):
                lap_times.append(state.lastLapTime)
                lap_count += 1
                lap_at_last_record = state.lap
                logger.info("Lap %d: %.3f s", lap_count, state.lastLapTime)
                if lap_count >= laps:
                    break

            # Limite di sicurezza: una policy che non riesce a completare un
            # giro (es. esce di pista e si ferma) altrimenti andrebbe in loop
            # per sempre, dato che il driver continua a inviare azioni e
            # TORCS non va mai in timeout. Aborta e registra il fallimento
            # invece di restare bloccato.
            if max_steps is not None and total_steps >= max_steps:
                logger.warning(
                    "Aborting after %d steps without completing %d lap(s).", total_steps, laps
                )
                aborted_no_lap = True
                break

    off_track_pct = (off_track_steps / max(total_steps, 1)) * 100.0
    avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 0.0
    max_speed = max(speed_samples) if speed_samples else 0.0

    results = {
        "driver": driver_name,
        "evaluated_at": datetime.now().isoformat(),
        "laps_requested": laps,
        "laps_completed": lap_count,
        "lap_times_s": lap_times,
        "best_lap_s": min(lap_times) if lap_times else None,
        "avg_lap_s": sum(lap_times) / len(lap_times) if lap_times else None,
        "max_speed_kmh": round(max_speed, 2),
        "avg_speed_kmh": round(avg_speed, 2),
        "off_track_pct": round(off_track_pct, 2),
        "damage": round(max_damage, 2),
        "total_steps": total_steps,
    }
    if max_steps is not None:
        results["aborted_no_lap"] = aborted_no_lap

    logger.info("Results: %s", json.dumps(results, indent=2))

    if output_path is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"eval_{driver_name}_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    return results
