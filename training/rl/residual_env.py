"""Ambiente Residual-RL: SAC impara una piccola correzione sopra al driver BC
funzionante, invece di provare a sostituirlo.

  azione_finale = base_bc.step(state) + RESIDUAL_SCALE * rl_residual

Il warm-start da una singola sotto-rete BC (train_sac.py senza --residual)
falliva: il driver che completa il giro è l'intera pipeline BC (fusione di
reti + guadagni + gestione marce), e l'esplorazione di SAC degradava la
sotto-rete fino a un'auto bloccata (0 giri completati).

`base_bc` è il blend BC legacy (LegacyBlendBCDriver, 121.978s), congelato in
drivers/rl/legacy_bc_blend.py perché `_DRIVER.driver.BCDriver` è cambiato da
allora (bc_tita_v20, poi cem_v5) e il residuo, addestrato su quella pipeline
specifica, non è mai stato validato su una base diversa.

`rl_residual` è in [-1, 1]^3 e l'attore SAC parte azzerato (zero_residual_actor):
il training inizia guidando come il driver BC e completa giri da subito;
l'RL impara poi correzioni piccole e limitate, che non possono far uscire di
pista o bloccare l'auto come la policy addestrata da zero.

Riusa TorcsSacEnv interamente, cambia solo come viene costruito il comando.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Base = blend BC legacy congelato, non il BCDriver principale attuale
# (vedi docstring del modulo e di drivers/rl/legacy_bc_blend.py).
from drivers.rl.legacy_bc_blend import LegacyBlendBCDriver
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.torcs_gym_env import TorcsSacEnv

# RESIDUAL_SCALE: magnitudine fisica massima della correzione RL su ciascuno
# di steer/accel/brake. Anche un ipotetico attacco "mai frenare" a scala
# 0.02-0.05 uscirebbe di pista solo in curva (circa 91% del percorso resta
# sicuro); una policy addestrata (che impara a frenare, con la penalità -200
# fuori pista) è al sicuro con 0.03. RESIDUAL_L2_COEF: penalità L2 sul
# residuo, così la policy corregge solo dove il guadagno in reward ripaga il
# costo, restando quasi puro BC altrove. Con questi valori il checkpoint
# consegnato (drivers/rl/models/sac_corkscrew_residual) valuta a 127.07s, 0%
# fuori pista (contro BC 121.978s/0%), circa 4% più lento di BC, perché
# l'obiettivo era un driver RL funzionante e sicuro, non battere il tempo
# giro di BC.
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
    """Azzera il layer di output dell'attore SAC: il residuo iniziale è 0
    (si parte guidando esattamente come la base BC) con std di esplorazione
    molto piccola (log_std=-3.0, std circa 0.05). Serve piccola perché il rumore si
    somma a un driver che già inserisce curve strette: una std più ampia
    (0.22 circa, provata in un run precedente) mandava l'auto fuori pista dopo
    285 step circa, prima di completare un giro, senza mai dare a SAC il segnale
    di giro completato. La policy deterministica mantiene comunque piena
    autorità ±RESIDUAL_SCALE, solo il rumore in training è ridotto.
    """
    actor = model.policy.actor
    with torch.no_grad():
        actor.mu.weight.zero_()
        actor.mu.bias.zero_()
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-3.0)
