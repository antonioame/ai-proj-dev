"""Ambiente Residual-RL: SAC impara una piccola correzione sopra al driver BC
funzionante, invece di provare a sostituirlo.

Perché esiste
-------------
Il primo tentativo RL (train_sac.py senza --residual) faceva il warm-start di
SAC da una singola sotto-rete BC grezza (bc_from_olddriver_v1) — ma il driver
che completa davvero il giro del Corkscrew è l'*intera* pipeline
`_DRIVER.driver.BCDriver`: una fusione di due reti, guadagni STEER/ACCEL/BRAKE
applicati a posteriori, gestione marce basata su RPM e una fase di avvio. Fare
il warm-start dai pesi di una singola sotto-rete NON è lo stesso che partire
dal driver funzionante, e l'esplorazione per entropia di SAC erodeva anche
quello fino a un'auto che si blocca (0 giri completati).

Il residual RL risolve entrambi i problemi:
  azione_finale = base_bc.step(state)  +  RESIDUAL_SCALE * rl_residual

dove `base_bc` è il blend BC legacy (LegacyBlendBCDriver, 121.978s) — la
pipeline completa su cui il checkpoint consegnato è stato addestrato. Nota
(2026-07-17): la base NON è più importata da `_DRIVER.driver.BCDriver`,
perché quel modulo dal 2026-07-15 contiene bc_tita_v20 (modello singolo,
gain diversi): il residuo sommato a una base diversa da quella di training
non è mai stato validato. La base legacy è congelata in
drivers/rl/legacy_bc_blend.py apposta per questo.
con `rl_residual` in [-1, 1]^3 e l'attore SAC azzerato all'init (vedi
zero_residual_actor), così all'inizio del training l'agente guida
*esattamente* come il driver BC da 121.978s e completa i giri da subito. L'RL
impara poi piccole correzioni limitate e dipendenti dallo stato per andare più
veloce. Essendo il residuo limitato, non può bloccarsi catastroficamente o
uscire di pista come faceva la policy addestrata da zero — la base BC tiene
l'auto in vita.

Questo riusa TorcsSacEnv interamente (rilancio TORCS per-episodio, reward,
terminazione, osservazione) e cambia solo il modo in cui viene costruito il
comando di controllo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Base = blend BC legacy, NON il BCDriver di produzione: il checkpoint
# residual consegnato è stato addestrato quando _DRIVER.driver.BCDriver era
# ancora il blend a due reti (pre-promozione bc_tita_v20 del 2026-07-15).
# Usare la base nuova invaliderebbe checkpoint esistente ed eventuali resume.
# Vedi il docstring di drivers/rl/legacy_bc_blend.py.
from drivers.rl.legacy_bc_blend import LegacyBlendBCDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.torcs_gym_env import TorcsSacEnv

# Magnitudine fisica massima della correzione RL su ciascuno di
# steer/accel/brake, e la penalità L2 che tiene il residuo addestrato vicino
# alla base BC.
#
# Insieme, questi due valori fanno sì che il driver residual consegnato
# funzioni davvero (completa il giro, 0% fuori pista) e sia genuinamente RL
# (una policy SAC addestrata corregge la base ad ogni step). Il checkpoint
# consegnato (drivers/rl/models/sac_corkscrew_residual) è stato addestrato con
# questi valori e valuta in modo deterministico a 127.07s, 0% fuori pista, 0
# danni (contro BC 121.978s / 0%).
#   * La scala limita quanto l'RL può allontanare l'auto dalla linea BC. Un
#     attacco "mai frenare" nel caso peggiore costante esce da una curva
#     (~91% del percorso) a ogni scala 0.02-0.05, ma è un attacco stupido che
#     non frena mai; una policy addestrata su giri puliti (con la penalità
#     -200 fuori pista) impara a frenare in curva, quindi 0.03 è sicuro per
#     una policy addestrata.
#   * RESIDUAL_L2_COEF penalizza ||residuo||^2 ad ogni step, quindi una
#     correzione deve ripagare il proprio costo in reward di guida — la
#     policy per default resta quasi puro BC e corregge solo dove aiuta
#     chiaramente. Con questo + un training per-episodio pulito (vedi il
#     commento sulla config SAC in train_sac sul perché il training per-step
#     corrompeva ogni run precedente) il residuo appreso mantiene l'auto
#     esattamente sulla linea di BC (0% fuori pista).
# Nota: questo NON batte BC sul tempo giro (~4% più lento) — l'obiettivo era
# un driver funzionante, genuinamente RL, che completasse il giro in
# sicurezza. Battere BC richiederebbe un reward basato sul tempo giro, non un
# aggiustamento della scala.
RESIDUAL_SCALE = 0.03
RESIDUAL_L2_COEF = 5.0


class ResidualTorcsSacEnv(TorcsSacEnv):
    """TorcsSacEnv in cui l'azione è un residuo limitato sopra al driver BC."""

    def __init__(self, *args, residual_scale: float = RESIDUAL_SCALE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._res_scale = residual_scale
        self._bc = LegacyBlendBCDriver()
        # L'attore SAC lavora in uno spazio di residuo normalizzato [-1, 1]^3;
        # la scala fisica viene applicata qui. La matematica standard della
        # target-entropy di SAC richiede [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Resetta il driver base BC così la sua fase di avvio e lo stato
        # marce ripartono con l'episodio; è lui a gestire il lancio (vedi
        # l'override di _run_startup).
        self._bc.on_restart()
        return super().reset(seed=seed, options=options)

    def _run_startup(self, state: SensorState) -> SensorState:
        # Nessun burst di avvio gestito dall'ambiente: il driver base BC
        # gestisce da solo la partenza da fermo nei suoi primi
        # BCDriver.STARTUP_STEPS step.
        return state

    def step(self, residual):
        self._ensure_started()  # lancio+connessione differiti al primo step (vedi env base)
        d = np.asarray(residual, dtype=np.float32)
        base = self._bc.step(self._last_state)
        cmd = Action(
            steer=base.steer + float(d[0]) * self._res_scale,
            accel=base.accel + float(d[1]) * self._res_scale,
            brake=base.brake + float(d[2]) * self._res_scale,
            gear=base.gear,
        )
        obs, reward, terminated, truncated, info = self._send_and_observe(cmd)
        # Penalizza lo scostamento dalla base BC (vedi RESIDUAL_L2_COEF).
        reward -= RESIDUAL_L2_COEF * float(np.sum(d * d))
        return obs, reward, terminated, truncated, info


def zero_residual_actor(model) -> None:
    """Azzera il layer di output dell'attore SAC così la media iniziale del
    residuo è 0 — cioè il training parte guidando esattamente come la base
    BC — e inizia con una std di esplorazione *molto* piccola.

    Qui la std di esplorazione conta molto: si somma a un driver BC che già
    inserisce le curve strette del Corkscrew, quindi anche un rumore di
    sterzo modesto (un tentativo precedente con std di esplorazione ≈0.22,
    sensibilmente più ampia dell'attuale) si accumula e manda l'auto fuori
    pista dopo ~285 step, prima ancora che completi un giro — così l'agente
    non vede mai il reward per aver finito. log_std = -3.0 (std≈0.05, che
    dopo la compressione tanh e la scala RESIDUAL_SCALE corrisponde a un
    jitter fisico di sterzo per step dell'ordine di qualche millesimo, con
    picchi fino a ~±0.007 su lunghe sequenze — verificato per simulazione)
    mantiene l'esplorazione abbastanza delicata da far restare l'auto
    in pista e far girare gli episodi per giri completi, dando a SAC un vero
    segnale di giro completato da cui imparare. La policy deterministica
    mantiene comunque la piena autorità ±RESIDUAL_SCALE — solo il rumore in
    fase di training è piccolo.
    """
    actor = model.policy.actor
    with torch.no_grad():
        actor.mu.weight.zero_()
        actor.mu.bias.zero_()
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-3.0)
