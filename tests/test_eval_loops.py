"""Test unitari per il rilevamento giri di evaluate_common.run_eval_loop.

Copre in particolare il caso limite corretto dal fix: la simulazione è
deterministica a parità di codice, quindi due giri consecutivi possono avere
lo stesso lastLapTime al millesimo, il conteggio deve comunque accorgersi
del nuovo giro grazie al contatore state.lap, senza contare due volte lo
stesso giro.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from torcs_env.actions import Action
from torcs_env.sensors import SensorState

import evaluate_common
from evaluate_common import run_eval_loop


class FakeDriver:
    """Driver fittizio: azione neutra, nessuno stato interno da resettare."""

    def step(self, state):
        return Action()

    def on_restart(self):
        pass


class FakeClient:
    """Sostituisce TORCSClient: restituisce in sequenza gli stati passati,
    poi SHUTDOWN. Nessuna connessione di rete reale."""

    def __init__(self, states, host=None, port=None):
        self._states = list(states) + [evaluate_common.SHUTDOWN]
        self._idx = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def receive(self):
        result = self._states[self._idx]
        self._idx += 1
        return result

    def send(self, action):
        pass


def _state(lastLapTime=0.0, lap=1, speed=100.0, damage=0.0, trackPos=0.0):
    return SensorState(lastLapTime=lastLapTime, lap=lap, speed=speed, damage=damage, trackPos=trackPos)


def _run(states, laps, tmp_path):
    with patch.object(evaluate_common, "TORCSClient", lambda host=None, port=None: FakeClient(states)):
        return run_eval_loop(
            FakeDriver(), "fake_driver", laps=laps, output_path=tmp_path / "result.json"
        )


def test_two_laps_different_times(tmp_path):
    states = [
        _state(lastLapTime=0.0, lap=1),
        _state(lastLapTime=50.123, lap=2),
        _state(lastLapTime=50.123, lap=2),  # stesso stato ripetuto: no doppio conteggio
        _state(lastLapTime=48.456, lap=3),
    ]
    result = _run(states, laps=2, tmp_path=tmp_path)
    assert result["laps_completed"] == 2
    assert result["lap_times_s"] == [50.123, 48.456]


def test_two_laps_identical_time_but_lap_advances(tmp_path):
    # Bug storico: tempi identici al millesimo su due giri consecutivi.
    # Il vecchio confronto solo su lastLapTime non vedeva il secondo giro;
    # state.lap che avanza da 1 a 2 deve comunque farlo contare.
    states = [
        _state(lastLapTime=0.0, lap=1),
        _state(lastLapTime=50.000, lap=2),
        _state(lastLapTime=50.000, lap=2),  # stesso lap, stesso tempo: nessun nuovo conteggio
        _state(lastLapTime=50.000, lap=3),  # nuovo giro, tempo identico, lap avanzato
    ]
    result = _run(states, laps=2, tmp_path=tmp_path)
    assert result["laps_completed"] == 2
    assert result["lap_times_s"] == [50.000, 50.000]


def test_no_double_count_on_constant_state(tmp_path):
    # laps=5 (irraggiungibile con questi dati): il loop deve proseguire fino
    # a SHUTDOWN senza mai ricontare il giro già registrato.
    states = [_state(lastLapTime=0.0, lap=1) for _ in range(5)]
    states += [_state(lastLapTime=50.0, lap=2) for _ in range(20)]
    result = _run(states, laps=5, tmp_path=tmp_path)
    assert result["laps_completed"] == 1
    assert result["lap_times_s"] == [50.0]
