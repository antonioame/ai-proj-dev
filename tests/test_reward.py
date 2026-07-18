"""Test unitari per le funzioni reward versionate della Fase 3 RL (non modifica reward.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from torcs_env.sensors import SensorState
from training.rl.reward import REWARD_VERSIONS


def test_baseline_v1_equals_speed_with_zero_angle_and_track_pos():
    prev_state = SensorState(speed=50.0, angle=0.0, trackPos=0.0, distRaced=0.0)
    state = SensorState(speed=50.0, angle=0.0, trackPos=0.0, distRaced=0.0)
    r = REWARD_VERSIONS["baseline_v1"].step_fn(prev_state, state)
    assert r == pytest.approx(50.0)


def test_refined_v2_adds_progress_term():
    prev_state = SensorState(speed=50.0, angle=0.0, trackPos=0.0, distRaced=100.0)
    state = SensorState(speed=50.0, angle=0.0, trackPos=0.0, distRaced=110.0)
    r = REWARD_VERSIONS["refined_v2"].step_fn(prev_state, state)
    # base (=50, angle/trackPos nulli) + 0.5 * (110-100)
    assert r == pytest.approx(50.0 + 0.5 * 10.0)


def test_refined_v2_standing_still_penalty():
    prev_state = SensorState(speed=2.0, angle=0.0, trackPos=0.0, distRaced=100.0)
    state = SensorState(speed=2.0, angle=0.0, trackPos=0.0, distRaced=100.0)
    r = REWARD_VERSIONS["refined_v2"].step_fn(prev_state, state)
    # base = speed = 2.0, progress = 0, penalità standing-still = -1.0
    assert r == pytest.approx(2.0 - 1.0)


def test_safe_progress_v4_stationary_car_has_no_positive_term_and_penalty():
    prev_state = SensorState(speed=0.0, angle=0.0, trackPos=0.0, distRaced=100.0)
    state = SensorState(speed=0.0, angle=0.0, trackPos=0.0, distRaced=100.0)
    r = REWARD_VERSIONS["safe_progress_v4"].step_fn(prev_state, state)
    # progress=0 (Δdist=0), control_penalty=0 (speed=0), unsafe_penalty=0 (speed=0),
    # standing_penalty=-1.0 → il reward totale è la sola penalità
    assert r == pytest.approx(-1.0)


def test_safe_progress_v4_negative_progress_is_clamped_to_zero():
    prev_state = SensorState(speed=50.0, angle=0.0, trackPos=0.0, distRaced=100.0)
    state = SensorState(speed=50.0, angle=0.0, trackPos=0.0, distRaced=90.0)  # distRaced calato
    r_negative = REWARD_VERSIONS["safe_progress_v4"].step_fn(prev_state, state)

    state_flat = SensorState(speed=50.0, angle=0.0, trackPos=0.0, distRaced=100.0)  # Δdist=0
    r_flat = REWARD_VERSIONS["safe_progress_v4"].step_fn(prev_state, state_flat)

    # Il termine di progresso è clampato a 0 in entrambi i casi (Δdist<0 e Δdist=0):
    # il reward deve essere identico.
    assert r_negative == pytest.approx(r_flat)
