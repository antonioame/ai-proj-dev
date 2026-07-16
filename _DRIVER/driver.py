"""Driver di Behavioral Cloning — modello singolo clonato dallo stile di guida
del bot nativo TORCS "tita" (bc_tita_v20), promosso a driver di produzione il
2026-07-15 al posto del precedente blend rettilineo/curva (bc_from_attempt1_v1
+ bc_from_olddriver_v1), verificato più lento (124.296s vs 111.986s, stesse
condizioni di test, entrambi puliti: 0% fuori pista, 0 danni).

I due modelli del vecchio blend restano in models/ (bc_from_attempt1_v1.*,
bc_from_olddriver_v1.*) per eventuale rollback, semplicemente non più caricati
da questa classe.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from drivers.bc_common import load_bc_model, shift_gear, startup_gear
from torcs_env.actions import Action
from torcs_env.sensors import SensorState
from training.rl.features import build_feature_vector


class BCDriver:
    """Driver BC a modello singolo, clonato dallo stile di guida di tita
    (bc_tita_v20: 13 giri puliti di telemetria del bot nativo + esempi di
    recupero raccolti con procedura DAgger-style, bc come rete di sicurezza).

    A differenza del precedente blend rettilineo/curva a due reti, questo è
    un unico BCPolicy che gestisce l'intero giro.
    """

    STEER_GAIN = 1.0   # Applicato all'output di sterzo — NON 1.8 (valore del
                        # vecchio blend): con questo modello, un gain più alto
                        # causava oscillazioni e uscite di pista in curva.
    ACCEL_GAIN = 1.40
    BRAKE_GAIN = 0.80

    # Fase di avvio: applica pieno gas con sterzo a zero per questo numero di step.
    # Mantiene l'auto dritta mentre il modello riceve input fuori distribuzione
    # (OOD: velocità≈0, marcia=0).
    STARTUP_STEPS = 80

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_gear = 1
        self._step_count = 0

        models_dir = Path(__file__).resolve().parent / "models"
        model_path = models_dir / "bc_tita_v20.pth"
        stats_path = models_dir / "bc_tita_v20.npz"

        for p in [model_path, stats_path]:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")

        self._model, self._mean, self._std = load_bc_model(model_path, stats_path, self._device)
        print("[BCDriver] Loaded checkpoint: bc_tita_v20 (clone dello stile di guida di tita)")

    def reset(self) -> None:
        pass

    def on_restart(self) -> None:
        self.current_gear = 1
        self._step_count = 0

    def step(self, state: SensorState) -> Action:
        self._step_count += 1

        if self._step_count <= self.STARTUP_STEPS:
            gear = startup_gear(state.speed)
            self.current_gear = gear
            return Action(steer=0.0, accel=1.0, brake=0.0, gear=gear).clamp()

        sensor_vec = build_feature_vector(state)

        t = torch.from_numpy(sensor_vec).float().to(self._device)
        t = (t - self._mean) / self._std
        with torch.no_grad():
            out = self._model(t)

        steer = max(-1.0, min(1.0, float(out["steer"].item()) * self.STEER_GAIN))
        accel = max(0.0, min(1.0, float(out["accel"].item()) * self.ACCEL_GAIN))
        brake = max(0.0, min(1.0, float(out["brake"].item()) * self.BRAKE_GAIN))

        self.current_gear = shift_gear(state.rpm, self.current_gear)

        return Action(
            steer=float(steer),
            accel=float(accel),
            brake=float(brake),
            gear=self.current_gear,
        ).clamp()
