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


# ---------------------------------------------------------------------------
# safe_progress_v3 — riprogettata dopo 3 run diretti indipendenti (auto entropy,
# ent_coef basso, ent_coef moderato con warmup corretto) che hanno TUTTI
# degenerato appena l'attore ha iniziato ad allenarsi davvero, sempre con lo
# stesso pattern: episodi via via più corti, schianti ripetuti nello stesso
# tornante, critic_loss in crescita senza freno. Diagnosi: in baseline_v1/
# refined_v2 il termine dominante è v*cos(angle) — sostanzialmente la velocità
# istantanea sommata per ogni singolo tick — mentre la penalità di uscita
# pista è un valore FISSO applicato una sola volta a fine episodio. Su un giro
# pulito (~6800 step) quella somma di velocità arriva a ~5*10^5; la penalità
# di -200 vale lo 0,04% del reward totale dell'episodio — un deterrente
# trascurabile. La policy impara semplicemente "vai più forte ovunque",
# indipendentemente da quanto sia rischioso in quel punto della pista, perché
# anche perdendo il controllo il reward di velocità già incassato nei tick
# precedenti supera quasi sempre la penalità fissa.
#
# safe_progress_v3 rimuove il termine di velocità istantanea come motore
# principale e lo sostituisce con:
#   1. Il progresso lungo il tracciato (distRaced) come UNICO termine positivo
#      dominante — lega il reward a ciò che conta davvero (percorrere la pista
#      velocemente), non alla velocità istantanea slegata dal contesto.
#   2. Una penalità di velocità-non-sicura: riusa la stessa formula fisica già
#      validata nel RuleBasedDriver isolato (physics_safe_speed, Sezione 2.2
#      di RELAZIONE_FINALE_V2.md — v_sicura = sqrt((distanza_libera-margine)
#      * BRAKE_DECEL_FACTOR)) per penalizzare la velocità che eccede quanto è
#      fisicamente frenabile in tempo dato il sensore frontale in quel preciso
#      istante — non un tetto di velocità fisso, ma "vai forte quando la
#      strada è libera, rallenta quando non lo è", esattamente il
#      comportamento che il driver rule-based e BC già dimostrano funzionare.
#   3. Una penalità di controllo leggera (heading/trackPos), proporzionale
#      alla velocità così non punisce una sbandata a passo d'uomo quanto una
#      ad alta velocità.
#   4. Penalità di uscita pista alzata a -500 — ma il punto chiave non è il
#      valore assoluto, è che ora è confrontata con un reward per giro molto
#      più piccolo (progresso ~3 * 3608m = 10824 su un giro pulito, non 5*10^5)
#      quindi pesa proporzionalmente molto di più (~4,6% del reward di un giro
#      pulito, contro lo 0,04% di refined_v2).
# ---------------------------------------------------------------------------

# Stessi valori di RuleBasedDriver (old_versions_drivers/project_V2/driver.py),
# calibrati e già validati su questo simulatore — non reinventati qui.
_BRAKE_DECEL_FACTOR = 270.0
_BRAKE_MARGIN = 5.0


def _fwd_dist(state: SensorState) -> float:
    """Minimo dei sensori ~-3°..+3° (indici 7-11 su 19) — stesso settore
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


# v3: prima validazione — ha dimostrato che la forma della reward è stabile
# (nessun crash, critic_loss limitato, auto-correzione dopo un'instabilità
# transitoria) ma con questi pesi è troppo prudente: 150,222s, punta max
# 121 km/h contro i 199,6 km/h di BC. Penalità di velocità-non-sicura e di
# controllo, entrambe proporzionali alla velocità, scoraggiano troppo la
# guida vicino al limite fisico invece di spingere la policy a starci sopra.
_safe_progress_step = _make_safe_progress_step(
    progress_weight=3.0, control_weight=0.03, unsafe_weight=0.4, safe_speed_scale=1.15,
)

# v4: retaratura dopo v3 — stessa forma (stabile, provata), pesi meno
# prudenti per spingere la policy più vicino al limite fisico:
#   - progress_weight alzato (più incentivo a coprire distanza rapidamente)
#   - control/unsafe_weight dimezzati o più (meno freno automatico sulla
#     velocità pura, che prima penalizzava la guida aggressiva anche quando
#     giustificata)
#   - safe_speed_scale alzato a 1.35, più vicino/oltre il TARGET_PHYSICS_SCALE
#     1.20 di RuleBasedDriver — BC supera anche quello grazie ai guadagni
#     STEER/ACCEL applicati a posteriori, quindi qui si dà più margine prima
#     che scatti qualunque penalità.
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
        # Alzata rispetto alla baseline per la Sezione 4.2.3: il vincolo
        # rigido del progetto è "nessuno schianto, escursioni fuori pista
        # minime", quindi la policy deve evitare con decisione le uscite di
        # pista invece di scambiare una piccola probabilità di uscita con
        # una velocità marginalmente più alta.
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
