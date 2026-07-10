"""Test unitari per l'interpretazione delle stringhe di sensori SCR."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from torcs_env.sensors import SensorState

# Una stringa di sensori SCR rappresentativa
SAMPLE = (
    "(angle 0.123)"
    "(speedX 87.5)"
    "(speedY -1.2)"
    "(speedZ 0.05)"
    "(trackPos -0.15)"
    "(track 200.0 198.5 150.3 120.0 95.0 80.0 70.0 65.0 62.0 60.0 "
    "62.0 65.0 70.0 80.0 95.0 120.0 150.3 198.5 200.0)"
    "(opponents 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 "
    "200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 "
    "200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 "
    "200.0 200.0 200.0 200.0 200.0)"
    "(rpm 5200.0)"
    "(gear 3)"
    "(damage 0.0)"
    "(distRaced 1234.5)"
    "(distFromStart 500.0)"
    "(lastLapTime 0.0)"
    "(curLapTime 18.3)"
    "(racePos 1)"
    "(fuel 90.0)"
    "(wheelSpinVel 150.0 151.0 149.5 150.5)"
    "(z 0.33)"
)


def test_angle():
    s = SensorState.from_string(SAMPLE)
    assert s.angle == pytest.approx(0.123)


def test_speed():
    s = SensorState.from_string(SAMPLE)
    assert s.speed == pytest.approx(87.5)


def test_speed_lateral():
    s = SensorState.from_string(SAMPLE)
    assert s.speedY == pytest.approx(-1.2)


def test_track_pos():
    s = SensorState.from_string(SAMPLE)
    assert s.trackPos == pytest.approx(-0.15)


def test_track_length():
    s = SensorState.from_string(SAMPLE)
    assert len(s.track) == 19


def test_track_centre_sensor():
    s = SensorState.from_string(SAMPLE)
    # L'indice 9 è dritto davanti
    assert s.track[9] == pytest.approx(60.0)


def test_opponents_length():
    s = SensorState.from_string(SAMPLE)
    assert len(s.opponents) == 36


def test_rpm():
    s = SensorState.from_string(SAMPLE)
    assert s.rpm == pytest.approx(5200.0)


def test_gear():
    s = SensorState.from_string(SAMPLE)
    assert s.gear == 3


def test_damage():
    s = SensorState.from_string(SAMPLE)
    assert s.damage == pytest.approx(0.0)


def test_dist_raced():
    s = SensorState.from_string(SAMPLE)
    assert s.distRaced == pytest.approx(1234.5)


def test_cur_lap_time():
    s = SensorState.from_string(SAMPLE)
    assert s.curLapTime == pytest.approx(18.3)


def test_fuel():
    s = SensorState.from_string(SAMPLE)
    assert s.fuel == pytest.approx(90.0)


def test_wheel_spin_vel_length():
    s = SensorState.from_string(SAMPLE)
    assert len(s.wheelSpinVel) == 4


def test_wheel_spin_vel_values():
    s = SensorState.from_string(SAMPLE)
    assert s.wheelSpinVel[0] == pytest.approx(150.0)
    assert s.wheelSpinVel[2] == pytest.approx(149.5)


def test_z():
    s = SensorState.from_string(SAMPLE)
    assert s.z == pytest.approx(0.33)


def test_default_lap_is_one():
    s = SensorState.from_string(SAMPLE)
    assert s.lap == 1


def test_raw_string_stored():
    s = SensorState.from_string(SAMPLE)
    assert s.raw == SAMPLE


def test_empty_string_returns_defaults():
    s = SensorState.from_string("")
    assert s.angle == 0.0
    assert len(s.track) == 19
    assert s.gear == 0
