"""Test unitari per le utility condivise dei driver BC (drivers/bc_common.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drivers.bc_common import shift_gear, startup_gear


def test_shift_gear_upshifts_above_threshold():
    assert shift_gear(rpm=12001.0, current_gear=3) == 4


def test_shift_gear_does_not_upshift_past_sixth():
    assert shift_gear(rpm=12001.0, current_gear=6) == 6


def test_shift_gear_downshifts_below_threshold():
    assert shift_gear(rpm=5999.0, current_gear=3) == 2


def test_shift_gear_does_not_downshift_past_first():
    assert shift_gear(rpm=5999.0, current_gear=1) == 1


def test_shift_gear_stays_same_within_thresholds():
    assert shift_gear(rpm=8000.0, current_gear=3) == 3


def test_startup_gear_first_below_15():
    assert startup_gear(speed=10.0) == 1


def test_startup_gear_second_between_15_and_45():
    assert startup_gear(speed=30.0) == 2


def test_startup_gear_third_above_45():
    assert startup_gear(speed=50.0) == 3
