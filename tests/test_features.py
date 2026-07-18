"""Test unitari per la costruzione del vettore di feature condiviso (Fase 3 RL)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from torcs_env.sensors import SensorState
from training.rl.features import FEATURE_DIM, build_feature_vector


def test_feature_dim_is_26():
    assert FEATURE_DIM == 26


def test_build_feature_vector_shape_and_dtype():
    state = SensorState()
    vec = build_feature_vector(state)
    assert vec.shape == (26,)
    assert vec.dtype == np.float32


def test_build_feature_vector_order():
    """Verifica l'ordine esatto [angle, speed, speedY, speedZ, trackPos, track(19), rpm, gear]
    usando valori tutti distinti per individuare eventuali scambi di posizione."""
    track = [10.0 + i for i in range(19)]
    state = SensorState(
        angle=1.0,
        speed=2.0,
        speedY=3.0,
        speedZ=4.0,
        trackPos=5.0,
        track=track,
        rpm=6000.0,
        gear=3,
    )
    vec = build_feature_vector(state)

    assert vec[0] == 1.0
    assert vec[1] == 2.0
    assert vec[2] == 3.0
    assert vec[3] == 4.0
    assert vec[4] == 5.0
    assert list(vec[5:24]) == track
    assert vec[24] == 6000.0
    assert vec[25] == 3.0
