"""Test unitari per la serializzazione di Action."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from torcs_env.actions import Action


def test_default_action_string():
    a = Action()
    s = a.to_string()
    assert "(accel 0.0000)" in s
    assert "(brake 0.0000)" in s
    assert "(steer 0.0000)" in s
    assert "(gear 1)" in s
    assert "(clutch 0.0000)" in s
    assert "(meta 0)" in s


def test_action_values_in_string():
    a = Action(steer=-0.5, accel=0.8, brake=0.0, gear=3, clutch=0.1)
    s = a.to_string()
    assert "(steer -0.5000)" in s
    assert "(accel 0.8000)" in s
    assert "(gear 3)" in s
    assert "(clutch 0.1000)" in s


def test_clamp_steer():
    a = Action(steer=2.5).clamp()
    assert a.steer == pytest.approx(1.0)

    a = Action(steer=-2.5).clamp()
    assert a.steer == pytest.approx(-1.0)


def test_clamp_accel():
    a = Action(accel=1.5).clamp()
    assert a.accel == pytest.approx(1.0)

    a = Action(accel=-0.5).clamp()
    assert a.accel == pytest.approx(0.0)


def test_clamp_brake():
    a = Action(brake=3.0).clamp()
    assert a.brake == pytest.approx(1.0)


def test_clamp_gear_max():
    a = Action(gear=9).clamp()
    assert a.gear == 6


def test_clamp_gear_min():
    a = Action(gear=-5).clamp()
    assert a.gear == -1


def test_clamp_does_not_mutate_original():
    original = Action(steer=3.0, accel=2.0)
    clamped = original.clamp()
    assert original.steer == 3.0  # invariato
    assert clamped.steer == 1.0


def test_reverse_gear():
    a = Action(gear=-1, accel=0.3)
    s = a.to_string()
    assert "(gear -1)" in s


def test_meta_flag():
    a = Action(meta=1)
    s = a.to_string()
    assert "(meta 1)" in s
