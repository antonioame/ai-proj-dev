"""Record keyboard-driven lap with real-time telemetry to CSV.

Workflow:
  Windows: launch TORCS in GUI mode (no -r flag)
           Start a race on Corkscrew
           Run this script to connect and record

This script:
  - Listens to keyboard input in real-time
  - Translates keypresses to steering/accel/brake commands
  - Sends actions to TORCS via SCR UDP
  - Records all sensors and actions to CSV
  - Detects lap completion and exits

Key bindings:
  W / ↑    : Accelerate
  S / ↓    : Brake
  A / ←    : Steer left
  D / →    : Steer right
  Q        : Downshift manually
  E        : Upshift manually
  (Gear is automatic based on RPM + manual override)

CSV output: data/keyboard_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pynput import keyboard
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient
from torcs_env.actions import Action

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _track_cols() -> list[str]:
    return [f"track_{i}" for i in range(19)]


FIELDNAMES = [
    "timestamp", "angle", "speed", "speedY", "speedZ", "trackPos",
    *_track_cols(),
    "rpm", "gear", "distRaced", "curLapTime",
    "steer", "accel", "brake", "gear_cmd",
]


class KeyboardController:
    """Manages keyboard input and translates to TORCS actions."""

    def __init__(self, auto_gear: bool = True):
        self.pressed_keys: set = set()
        self.lock = Lock()
        self.current_gear = 1
        self.auto_gear = auto_gear

        # Steering: how much to steer per key press
        self.steer_scale = 0.5

        # Accel/brake intensity
        self.accel_scale = 0.8
        self.brake_scale = 0.8

        # Auto gear thresholds (RPM) with hysteresis
        self.gear_up_rpm = 7500  # Reduced to shift up earlier
        self.gear_down_rpm = 2500  # Reduced to shift down more aggressively
        self.last_gear_change_step = -100  # Avoid multiple changes per step

    def on_press(self, key):
        try:
            char = key.char
            if char:
                with self.lock:
                    self.pressed_keys.add(char.lower())
        except AttributeError:
            # Special keys (arrows, etc)
            with self.lock:
                self.pressed_keys.add(key)

    def on_release(self, key):
        try:
            char = key.char
            if char:
                with self.lock:
                    self.pressed_keys.discard(char.lower())
        except AttributeError:
            with self.lock:
                self.pressed_keys.discard(key)

    def get_action(self, state, step: int = 0) -> Action:
        """Return current Action based on pressed keys and current state."""
        with self.lock:
            keys = self.pressed_keys.copy()

        # Sync with actual gear from TORCS (to stay in sync)
        self.current_gear = state.gear

        steer = 0.0
        accel = 0.0
        brake = 0.0

        # Steering: -1 (right) to +1 (left)
        if 'a' in keys or keyboard.Key.left in keys:
            steer += self.steer_scale
        if 'd' in keys or keyboard.Key.right in keys:
            steer -= self.steer_scale

        # Accel / Brake
        if 'w' in keys or keyboard.Key.up in keys:
            accel = self.accel_scale
        if 's' in keys or keyboard.Key.down in keys:
            brake = self.brake_scale

        # Manual gear control (Q=down, E=up)
        if (step - self.last_gear_change_step) >= 5:
            if 'q' in keys and self.current_gear > 1:
                self.current_gear -= 1
                self.last_gear_change_step = step
            elif 'e' in keys and self.current_gear < 6:
                self.current_gear += 1
                self.last_gear_change_step = step

        # Auto gear management with cooldown (min 5 steps ~250ms between changes)
        if self.auto_gear and (step - self.last_gear_change_step) >= 5:
            if state.rpm > self.gear_up_rpm and self.current_gear < 6:
                self.current_gear += 1
                self.last_gear_change_step = step
            elif state.rpm < self.gear_down_rpm and self.current_gear > 1:
                self.current_gear -= 1
                self.last_gear_change_step = step

        action = Action(
            steer=steer,
            accel=accel,
            brake=brake,
            gear=self.current_gear,
        )
        return action.clamp()

    def start_listening(self):
        """Start background keyboard listener."""
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        logger.info("Keyboard listener started.")

    def stop_listening(self):
        """Stop background keyboard listener."""
        if self.listener:
            self.listener.stop()
            logger.info("Keyboard listener stopped.")


def record(host: str | None = None, port: int | None = None) -> Path:
    """Record one keyboard-driven lap with telemetry."""
    controller = KeyboardController()
    controller.start_listening()

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"keyboard_{timestamp_str}.csv"

    rows: list[dict] = []
    lap_start: float | None = None
    lap_completed = False
    lap_complete_step = -1
    step = 0

    try:
        with TORCSClient(host=host, port=port) as client:
            logger.info("Connected to TORCS. Recording keyboard-driven lap.")
            logger.info("Controls: W/↑=accel, S/↓=brake, A/←=steer-left, D/→=steer-right")
            logger.info("Gear: auto (2500-7500 RPM), or Q=down-shift, E=up-shift")

            while not lap_completed:
                result = client.receive()

                if result == SHUTDOWN:
                    logger.info("Server shutdown.")
                    break
                if result == RESTART:
                    logger.info("Race restarted.")
                    rows.clear()
                    lap_start = None
                    lap_complete_step = -1
                    step = 0
                    continue

                state = result
                action = controller.get_action(state, step=step)
                client.send(action)
                step += 1

                now = time.time()
                if lap_start is None:
                    lap_start = now

                row = {
                    "timestamp": now,
                    "angle": state.angle,
                    "speed": state.speed,
                    "speedY": state.speedY,
                    "speedZ": state.speedZ,
                    "trackPos": state.trackPos,
                    **{f"track_{i}": state.track[i] for i in range(min(19, len(state.track)))},
                    "rpm": state.rpm,
                    "gear": state.gear,
                    "distRaced": state.distRaced,
                    "curLapTime": state.curLapTime,
                    "steer": action.steer,
                    "accel": action.accel,
                    "brake": action.brake,
                    "gear_cmd": action.gear,
                }
                rows.append(row)

                # Live status every ~1 second (20 steps @ 50ms each)
                if len(rows) % 20 == 0:
                    logger.info(
                        "time=%.1f speed=%.1f gear=%d trackPos=%.2f angle=%.2f "
                        "steer=%.2f accel=%.2f brake=%.2f",
                        state.curLapTime, state.speed, state.gear, state.trackPos,
                        state.angle, action.steer, action.accel, action.brake
                    )

                # Detect lap completion
                if state.lastLapTime > 0 and rows and lap_complete_step < 0:
                    lap_time = state.lastLapTime
                    logger.info("✓ Lap completed in %.3f s", lap_time)
                    lap_complete_step = step

                # Wait 20 more steps (~1 second) after lap completion before exiting
                # This gives TORCS time to stabilize before closing
                if lap_complete_step >= 0 and (step - lap_complete_step) >= 20:
                    lap_completed = True

    except Exception as e:
        logger.warning("Connection lost: %s (this is normal if TORCS closed after lap)", e)
    finally:
        controller.stop_listening()

    # Write CSV
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows to %s", len(rows), out_path)
    return out_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Record keyboard-driven lap with telemetry")
    parser.add_argument("--host", default=None, help="TORCS server host (default: localhost)")
    parser.add_argument("--port", type=int, default=None, help="TORCS server port (default: 3001)")
    args = parser.parse_args()
    record(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
