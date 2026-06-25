"""Unit tests for TORCSClient — all network calls are mocked."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from torcs_env.client import TORCSClient, RESTART, SHUTDOWN
from torcs_env.sensors import SensorState
from torcs_env.actions import Action

# A minimal valid sensor string
_SENSOR = (
    b"(angle 0.0)(speedX 50.0)(speedY 0.0)(speedZ 0.0)(trackPos 0.0)"
    b"(track 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200)"
    b"(opponents 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 "
    b"200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200)"
    b"(rpm 4000)(gear 2)(damage 0)(distRaced 100)(distFromStart 100)"
    b"(lastLapTime 0)(curLapTime 5.0)(racePos 1)(fuel 90)(wheelSpinVel 100 100 100 100)(z 0.3)"
)


@patch("torcs_env.client.socket.socket")
def test_handshake_sends_init_message(mock_socket_cls):
    """connect() must send the SCR init string."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recvfrom.return_value = (_SENSOR, ("127.0.0.1", 3001))

    client = TORCSClient(host="localhost", port=3001)
    client.connect()

    sent_data = mock_sock.sendto.call_args[0][0]
    assert sent_data.startswith(b"SCR(init ")
    assert b"0" in sent_data  # contains the 0-degree centre angle


@patch("torcs_env.client.socket.socket")
def test_receive_returns_sensor_state(mock_socket_cls):
    """receive() returns a SensorState for a normal sensor packet."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    # First call: handshake; second: receive
    mock_sock.recvfrom.side_effect = [
        (_SENSOR, ("127.0.0.1", 3001)),
        (_SENSOR, ("127.0.0.1", 3001)),
    ]

    client = TORCSClient(host="localhost", port=3001)
    client.connect()
    result = client.receive()

    assert isinstance(result, SensorState)
    assert result.speed == pytest.approx(50.0)


@patch("torcs_env.client.socket.socket")
def test_receive_returns_restart_sentinel(mock_socket_cls):
    """receive() returns RESTART when the server sends ***restart***."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recvfrom.side_effect = [
        (_SENSOR, ("127.0.0.1", 3001)),        # handshake
        (b"***restart***", ("127.0.0.1", 3001)),  # restart signal
    ]

    client = TORCSClient()
    client.connect()
    result = client.receive()
    assert result == RESTART


@patch("torcs_env.client.socket.socket")
def test_receive_returns_shutdown_sentinel(mock_socket_cls):
    """receive() returns SHUTDOWN when the server sends ***shutdown***."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recvfrom.side_effect = [
        (_SENSOR, ("127.0.0.1", 3001)),
        (b"***shutdown***", ("127.0.0.1", 3001)),
    ]

    client = TORCSClient()
    client.connect()
    result = client.receive()
    assert result == SHUTDOWN


@patch("torcs_env.client.socket.socket")
def test_send_encodes_action_string(mock_socket_cls):
    """send() encodes the action to the expected wire format."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recvfrom.return_value = (_SENSOR, ("127.0.0.1", 3001))

    client = TORCSClient()
    client.connect()
    action = Action(steer=0.25, accel=0.7, brake=0.0, gear=2)
    client.send(action)

    # The last sendto call (after the handshake sendto) carries the action
    last_sent = mock_sock.sendto.call_args_list[-1][0][0]
    assert b"(steer 0.2500)" in last_sent
    assert b"(accel 0.7000)" in last_sent
    assert b"(gear 2)" in last_sent


@patch("torcs_env.client.socket.socket")
def test_close_releases_socket(mock_socket_cls):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recvfrom.return_value = (_SENSOR, ("127.0.0.1", 3001))

    client = TORCSClient()
    client.connect()
    client.close()

    mock_sock.close.assert_called_once()


@patch("torcs_env.client.socket.socket")
def test_context_manager_closes_on_exit(mock_socket_cls):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recvfrom.return_value = (_SENSOR, ("127.0.0.1", 3001))

    with TORCSClient():
        pass

    mock_sock.close.assert_called_once()


@patch("torcs_env.client.socket.socket")
def test_lap_counter_increments_on_dist_raced_reset(mock_socket_cls):
    """Lap counter increments when distRaced drops (server reset)."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock

    def sensor(dist_raced: float) -> bytes:
        return (
            f"(angle 0)(speedX 50)(trackPos 0)(rpm 4000)(gear 2)"
            f"(distRaced {dist_raced})(distFromStart 0)(curLapTime 0)"
            f"(lastLapTime 0)(racePos 1)(damage 0)(fuel 90)(z 0)"
            f"(track 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200)"
            f"(opponents 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200"
            f" 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200)"
            f"(wheelSpinVel 100 100 100 100)(speedY 0)(speedZ 0)"
        ).encode()

    mock_sock.recvfrom.side_effect = [
        (sensor(5000.0), ("127.0.0.1", 3001)),   # handshake reply
        (sensor(5000.0), ("127.0.0.1", 3001)),   # normal step (lap 1)
        (sensor(10.0),   ("127.0.0.1", 3001)),   # distRaced reset → lap 2
    ]

    client = TORCSClient()
    client.connect()

    s1 = client.receive()
    assert isinstance(s1, SensorState)
    assert s1.lap == 1

    s2 = client.receive()
    assert isinstance(s2, SensorState)
    assert s2.lap == 2
