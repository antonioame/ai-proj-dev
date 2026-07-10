"""Funzioni reward versionate per la Fase 3 RL (REINFORCEMENT_LEARNING.md Sezione 4).

Ogni variante è una RewardVersion autonoma, così un run di training può
registrare esattamente quale formula ha prodotto i suoi risultati (Sezione
4.2: "log the reward formula version used for every training run").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from torcs_env.sensors import SensorState

StepRewardFn = Callable[[SensorState, SensorState], float]


@dataclass(frozen=True)
class RewardVersion:
    name: str
    off_track_penalty: float
    lap_bonus: float
    step_fn: StepRewardFn


def _baseline_step(prev_state: SensorState, state: SensorState) -> float:
    """Baseline di Sezione 4.1: r = v*cos(angle) - v*|sin(angle)| - v*|trackPos|."""
    v = state.speed
    return v * math.cos(state.angle) - v * abs(math.sin(state.angle)) - v * abs(state.trackPos)


# Il peso mantiene il termine di progresso all'incirca sullo stesso ordine di
# grandezza dei termini di velocità sopra (Sezione 4.2.5, "normalization") —
# i delta di distRaced per step a 50 Hz sono dell'ordine di pochi metri a
# velocità di gara, lo stesso ordine di v*cos.
_PROGRESS_WEIGHT = 0.5
_STANDING_STILL_KMH = 5.0
_STANDING_STILL_PENALTY = -1.0


def _refined_step(prev_state: SensorState, state: SensorState) -> float:
    """Perfezionamenti di Sezione 4.2 rispetto a baseline_v1:

    1. Reward di progresso — proporzionale al distRaced percorso in questo
       step, non solo alla velocità istantanea. Punta alla vera metrica di
       successo (un giro completo e veloce) ed è la mitigazione del corso
       stesso contro il reward hacking (Sezione 4.3).
    2. Guardia standing-still — penalità fissa sotto _STANDING_STILL_KMH,
       così stare fermi/girare sul posto non può superare in punteggio una
       guida attenta.

    La severità di terminazione fuori pista (perfezionamento 3) risiede in
    RewardVersion.off_track_penalty, applicata dal chiamante alla terminazione.
    """
    base = _baseline_step(prev_state, state)
    progress = _PROGRESS_WEIGHT * max(0.0, state.distRaced - prev_state.distRaced)
    standing_penalty = _STANDING_STILL_PENALTY if state.speed < _STANDING_STILL_KMH else 0.0
    return base + progress + standing_penalty


REWARD_VERSIONS: dict[str, RewardVersion] = {
    "baseline_v1": RewardVersion(
        name="baseline_v1",
        off_track_penalty=-100.0,
        lap_bonus=50.0,
        step_fn=_baseline_step,
    ),
    "refined_v2": RewardVersion(
        name="refined_v2",
        # Alzata rispetto alla baseline per la Sezione 4.2.3: il vincolo
        # rigido del progetto è "nessuno schianto, escursioni fuori pista
        # minime", quindi la policy deve evitare con decisione le uscite di
        # pista invece di scambiare una piccola probabilità di uscita con
        # una velocità marginalmente più alta.
        off_track_penalty=-200.0,
        lap_bonus=100.0,
        step_fn=_refined_step,
    ),
}
