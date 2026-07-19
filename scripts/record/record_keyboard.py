"""Registra un giro guidato da tastiera con telemetria in tempo reale su CSV.

Workflow:
  Windows: avvia TORCS in modalità GUI (senza flag -r)
           Avvia una gara sul Corkscrew
           Esegui questo script per connetterti e registrare

Questo script:
  - Ascolta l'input da tastiera in tempo reale
  - Traduce le pressioni dei tasti in comandi di sterzo/accelerazione/frenata
  - Invia le azioni a TORCS via UDP SCR
  - Registra tutti i sensori e le azioni su CSV
  - Rileva il completamento del giro ed esce

Associazioni tasti:
  W / ↑    : Accelera
  S / ↓    : Frena
  A / ←    : Sterza a sinistra
  D / →    : Sterza a destra
  Q        : Scala marcia (solo controllo manuale)
  E        : Sale di marcia (solo controllo manuale)
  (Il cambio marcia è manuale — hai il controllo completo)

Output CSV: data/keyboard_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pynput import keyboard
from telemetry_row import TRACK_COLS, build_row
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient
from torcs_env.actions import Action

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


FIELDNAMES = [
    "timestamp", "angle", "speed", "speedY", "speedZ", "trackPos",
    *TRACK_COLS,
    "rpm", "gear", "distRaced", "curLapTime",
    "steer", "accel", "brake", "gear_cmd",
]


class KeyboardController:
    """Gestisce l'input da tastiera e lo traduce in azioni TORCS."""

    def __init__(self, auto_gear: bool = False):
        self.pressed_keys: set = set()
        self.lock = Lock()
        self.current_gear = 1
        self.listener = None  # creato in start_listening(); evita AttributeError in stop_listening()
        self.auto_gear = auto_gear  # Disabilitato di default (solo manuale)

        # Sterzo: quanto sterzare per ogni pressione tasto
        self.steer_scale = 0.5

        # Intensità accelerazione/frenata
        self.accel_scale = 0.8
        self.brake_scale = 0.8

        # Cooldown cambio marcia (0.3s = ~15 step a 50Hz)
        self.gear_cooldown_steps = 15
        self.last_gear_change_step = -100

    def on_press(self, key):
        try:
            char = key.char
            if char:
                with self.lock:
                    self.pressed_keys.add(char.lower())
        except AttributeError:
            # Tasti speciali (frecce, ecc.)
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
        """Restituisce l'Action corrente in base ai tasti premuti e allo stato attuale."""
        with self.lock:
            keys = self.pressed_keys.copy()

        # Sincronizza con la marcia reale da TORCS (per restare allineati)
        self.current_gear = state.gear

        steer = 0.0
        accel = 0.0
        brake = 0.0

        # Sterzo: -1 (destra) a +1 (sinistra)
        if 'a' in keys or keyboard.Key.left in keys:
            steer += self.steer_scale
        if 'd' in keys or keyboard.Key.right in keys:
            steer -= self.steer_scale

        # Accelerazione / Frenata
        if 'w' in keys or keyboard.Key.up in keys:
            accel = self.accel_scale
        if 's' in keys or keyboard.Key.down in keys:
            brake = self.brake_scale

        # Controllo manuale marcia (Q=scala, E=sale) con cooldown
        # Cooldown = 15 step = ~0.3 secondi a 50Hz (finestra di debounce tarata empiricamente)
        if (step - self.last_gear_change_step) >= self.gear_cooldown_steps:
            if 'q' in keys and self.current_gear > 1:
                self.current_gear -= 1
                self.last_gear_change_step = step
            elif 'e' in keys and self.current_gear < 6:
                self.current_gear += 1
                self.last_gear_change_step = step

        action = Action(
            steer=steer,
            accel=accel,
            brake=brake,
            gear=self.current_gear,
        )
        return action.clamp()

    def start_listening(self):
        """Avvia il listener da tastiera in background."""
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        logger.info("Keyboard listener started.")

    def stop_listening(self):
        """Ferma il listener da tastiera in background."""
        if self.listener:
            self.listener.stop()
            logger.info("Keyboard listener stopped.")


def record(host: str | None = None, port: int | None = None) -> Path:
    """Registra un giro guidato da tastiera con telemetria."""
    controller = KeyboardController()
    controller.start_listening()

    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"keyboard_{timestamp_str}.csv"

    rows: list[dict] = []
    lap_completed = False
    lap_complete_step = -1
    step = 0

    try:
        with TORCSClient(host=host, port=port) as client:
            logger.info("Connected to TORCS. Recording keyboard-driven lap.")
            logger.info("Controls: W/↑=accel, S/↓=brake, A/←=steer-left, D/→=steer-right")
            logger.info("Gear: MANUAL CONTROL — Q=downshift, E=upshift (you have full control)")

            while not lap_completed:
                result = client.receive()

                if result == SHUTDOWN:
                    logger.info("Server shutdown.")
                    break
                if result == RESTART:
                    logger.info("Race restarted.")
                    rows.clear()
                    lap_complete_step = -1
                    step = 0
                    continue

                state = result
                action = controller.get_action(state, step=step)
                client.send(action)
                step += 1

                rows.append(build_row(time.time(), state, action))

                # Stato live ogni 20 step (~0.4s a 50 step/s)
                if len(rows) % 20 == 0:
                    logger.info(
                        "time=%.1f speed=%.1f gear=%d trackPos=%.2f angle=%.2f "
                        "steer=%.2f accel=%.2f brake=%.2f",
                        state.curLapTime, state.speed, state.gear, state.trackPos,
                        state.angle, action.steer, action.accel, action.brake
                    )

                # Rileva il completamento del giro
                if state.lastLapTime > 0 and rows and lap_complete_step < 0:
                    lap_time = state.lastLapTime
                    logger.info("✓ Lap completed in %.3f s", lap_time)
                    lap_complete_step = step

                # Attende altri 20 step (~0.4s a 50 step/s) dopo il completamento
                # del giro prima di uscire. Dà a TORCS il tempo di stabilizzarsi
                # prima della chiusura
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
