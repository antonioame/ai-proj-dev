"""Funzioni reward versionate per la Fase 3 RL.

Ogni variante è una RewardVersion autonoma, così un run di training può
registrare esattamente quale formula ha prodotto i suoi risultati.
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
    """Baseline: r = v*cos(angle) - v*|sin(angle)| - v*|trackPos|."""
    v = state.speed
    return v * math.cos(state.angle) - v * abs(math.sin(state.angle)) - v * abs(state.trackPos)


# Normalizza il progresso sullo stesso ordine di grandezza di v*cos sopra.
_PROGRESS_WEIGHT = 0.5
_STANDING_STILL_KMH = 5.0
_STANDING_STILL_PENALTY = -1.0


def _refined_step(prev_state: SensorState, state: SensorState) -> float:
    """Rispetto a baseline_v1: aggiunge reward di progresso (distRaced, non solo
    velocità istantanea, per mitigare reward hacking) e una penalità fissa sotto
    _STANDING_STILL_KMH per non premiare lo stare fermi/girare sul posto."""
    base = _baseline_step(prev_state, state)
    progress = _PROGRESS_WEIGHT * max(0.0, state.distRaced - prev_state.distRaced)
    standing_penalty = _STANDING_STILL_PENALTY if state.speed < _STANDING_STILL_KMH else 0.0
    return base + progress + standing_penalty


# safe_progress_v3: baseline_v1/refined_v2 degeneravano sempre allo stesso modo
# durante il training (episodi più corti, schianti ripetuti, critic_loss in
# crescita) perché il reward di velocità istantanea si accumula ad ogni tick
# mentre la penalità di uscita pista è fissa e una tantum: sommata su un giro
# pulito, quella penalità pesa lo 0,04% del reward totale, un deterrente
# trascurabile. v3 sostituisce la velocità istantanea col progresso lungo il
# tracciato come termine dominante, e aggiunge una penalità di
# velocità-non-sicura basata sulla stessa fisica già validata nel
# RuleBasedDriver (v_sicura = sqrt((distanza_libera-margine)*BRAKE_DECEL_FACTOR)):
# "vai forte quando la strada è libera, rallenta quando non lo è". La penalità
# di uscita pista sale a -500, che ora pesa circa il 4,6% del reward di un giro pulito.

# Stessi valori di RuleBasedDriver (old_versions_drivers/project_V2/driver.py).
_BRAKE_DECEL_FACTOR = 270.0
_BRAKE_MARGIN = 5.0


def _fwd_dist(state: SensorState) -> float:
    """Minimo dei sensori da -3° a +3° circa (indici 7-11 su 19), stesso settore
    frontale usato da RuleBasedDriver._fwd_dist, per coerenza."""
    track = state.track
    if len(track) < 12:
        return 200.0
    return min(track[7:12])


def _physics_safe_speed(fwd_dist: float) -> float:
    return math.sqrt(max(0.0, (fwd_dist - _BRAKE_MARGIN) * _BRAKE_DECEL_FACTOR))


def _make_safe_progress_step(
    progress_weight: float,
    control_weight: float,
    unsafe_weight: float,
    safe_speed_scale: float,
) -> StepRewardFn:
    def step_fn(prev_state: SensorState, state: SensorState) -> float:
        progress = progress_weight * max(0.0, state.distRaced - prev_state.distRaced)

        control_penalty = control_weight * state.speed * (
            abs(math.sin(state.angle)) + abs(state.trackPos)
        )

        safe_speed = _physics_safe_speed(_fwd_dist(state)) * safe_speed_scale
        unsafe_excess = max(0.0, state.speed - safe_speed)
        unsafe_penalty = unsafe_weight * unsafe_excess

        standing_penalty = _STANDING_STILL_PENALTY if state.speed < _STANDING_STILL_KMH else 0.0

        return progress - control_penalty - unsafe_penalty + standing_penalty

    return step_fn


# v3: forma stabile ma troppo prudente (150,222s, max 121 km/h vs 199,6 km/h di BC).
_safe_progress_step = _make_safe_progress_step(
    progress_weight=3.0, control_weight=0.03, unsafe_weight=0.4, safe_speed_scale=1.15,
)

# v4: stessa forma di v3, pesi meno prudenti (più progress_weight, meno
# control/unsafe_weight, safe_speed_scale più alto) per avvicinare la policy
# al limite fisico invece di tenerla indietro.
_safe_progress_step_v4 = _make_safe_progress_step(
    progress_weight=4.0, control_weight=0.015, unsafe_weight=0.15, safe_speed_scale=1.35,
)


REWARD_VERSIONS: dict[str, RewardVersion] = {
    "baseline_v1": RewardVersion(
        name="baseline_v1",
        off_track_penalty=-100.0,
        lap_bonus=50.0,
        step_fn=_baseline_step,
    ),
    "refined_v2": RewardVersion(
        name="refined_v2",
        # Penalità alzata: il vincolo del progetto è "nessuno schianto", quindi
        # niente compromessi fra rischio di uscita e velocità marginale.
        off_track_penalty=-200.0,
        lap_bonus=100.0,
        step_fn=_refined_step,
    ),
    "safe_progress_v3": RewardVersion(
        name="safe_progress_v3",
        off_track_penalty=-500.0,
        lap_bonus=200.0,
        step_fn=_safe_progress_step,
    ),
    "safe_progress_v4": RewardVersion(
        name="safe_progress_v4",
        off_track_penalty=-500.0,
        lap_bonus=200.0,
        step_fn=_safe_progress_step_v4,
    ),
}
