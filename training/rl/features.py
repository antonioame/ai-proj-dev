"""Costruzione condivisa del vettore di feature per la Fase 3 RL.

Questo è l'unico punto in cui l'osservazione a 26 dimensioni viene costruita
a partire da un SensorState, usato sia da training/rl/torcs_gym_env.py sia da
drivers/rl/driver.py. Tenerlo in un'unica funzione (invece di duplicarlo, come
faceva il precedente tentativo RL) protegge esattamente dal bug che affondò
quel tentativo: lo spazio di osservazione RL che derivava silenziosamente
fuori sincrono rispetto a ciò su cui la rete BC era realmente addestrata
(vedi la git history intorno ai commit 727593b / d357744 / 074c1ee per
l'incidente — la BC usava [speed, trackPos, angle, rpm, gear, track[6],
track[12], track[18]] mentre l'RL usava indici track diversi, causando un
bug persistente di sterzo-zero mai risolto del tutto prima che l'intero
tentativo di Fase 3 fosse rimosso).

Il layout corrisponde esattamente al sensor_vec di _DRIVER/driver.py:
    [angle, speed, speedY, speedZ, trackPos, *track(19), rpm, gear]
"""

from __future__ import annotations

import numpy as np

from torcs_env.sensors import SensorState

FEATURE_DIM = 26


def build_feature_vector(state: SensorState) -> np.ndarray:
    return np.array(
        [
            state.angle,
            state.speed,
            state.speedY,
            state.speedZ,
            state.trackPos,
            *state.track,
            state.rpm,
            float(state.gear),
        ],
        dtype=np.float32,
    )
