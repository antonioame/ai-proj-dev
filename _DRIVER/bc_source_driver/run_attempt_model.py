"""
Source driver used to generate the telemetry that trained _DRIVER/models/bc_from_attempt1_v1.
This is an earlier driving-net attempt, kept only to regenerate training samples.
Run this again if you need fresh samples to retrain that BC model.

Usage:
    conda run -n ai_env python _DRIVER/bc_source_driver/run_attempt_model.py [--host localhost] [--port 3001]

The model must be pre-trained:
    conda run -n ai_env python _DRIVER/bc_source_driver/train_attempt_model.py --csv data/<telemetry>.csv
"""

import sys
import csv
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "attempt_model"))

import snakeoil3_jm2 as snakeoil3


class DrivingNet(nn.Module):
    """Earlier driving-net attempt (precursor to the BC straight-line model)."""
    def __init__(self, dim_ingresso: int, numero_marce: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim_ingresso, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head_steer = nn.Linear(64, 1)
        self.head_accel_brake = nn.Linear(64, 2)
        self.head_gear = nn.Linear(64, numero_marce)

    def forward(self, dati_ingresso):
        strato_nascosto = self.backbone(dati_ingresso)
        return (
            torch.tanh(self.head_steer(strato_nascosto)),
            torch.sigmoid(self.head_accel_brake(strato_nascosto)),
            self.head_gear(strato_nascosto),
        )


def build_input_vector(sensori_stato, colonne_selezionate):
    """Build normalized input vector from sensors (TORCS format)."""
    distanze_tracciato = sensori_stato.get("track", [200.0] * 19)
    rotazione_ruote = sensori_stato.get("wheelSpinVel", [0.0] * 4)

    if len(distanze_tracciato) != 19:
        distanze_tracciato = [200.0] * 19
    if len(rotazione_ruote) != 4:
        rotazione_ruote = [0.0] * 4

    dizionario_caratteristiche = {}
    for i in range(19):
        dizionario_caratteristiche[f"track_{i}"] = float(distanze_tracciato[i])
    for i in range(4):
        dizionario_caratteristiche[f"wheel_{i}"] = float(rotazione_ruote[i])

    # Map TORCS speedX to 'speed' (as in our CSV training data)
    dizionario_caratteristiche["speed"] = float(sensori_stato.get("speedX", 0))
    dizionario_caratteristiche["trackPos"] = float(sensori_stato.get("trackPos", 0))
    dizionario_caratteristiche["angle"] = float(sensori_stato.get("angle", 0))
    dizionario_caratteristiche["rpm"] = float(sensori_stato.get("rpm", 0))

    return np.array([dizionario_caratteristiche[colonna] for colonna in colonne_selezionate], dtype=np.float32)


def main(host: str = "localhost", port: int = 3001):
    model_dir = Path(__file__).resolve().parent / "attempt_model"
    model_path = model_dir / "driving_model.pt"
    scaler_path = model_dir / "driving_scaler.pkl"

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Run: conda run -n ai_env python _DRIVER/bc_source_driver/train_attempt_model.py --csv data/<telemetry>.csv")
        sys.exit(1)

    if not scaler_path.exists():
        print(f"[ERROR] Scaler not found: {scaler_path}")
        sys.exit(1)

    # Load scaler
    print("[INFO] Loading model and scaler...")
    dati_scaler = joblib.load(scaler_path)
    media_normalizzazione = dati_scaler["mean"]
    deviazione_standard = dati_scaler["std"]
    colonne_input = dati_scaler["input_cols"]
    offset_marce = dati_scaler["gear_offset"]

    dispositivo = torch.device("cpu")
    modello_guida = DrivingNet(dim_ingresso=len(colonne_input)).to(dispositivo)
    modello_guida.load_state_dict(torch.load(model_path, map_location=dispositivo))
    modello_guida.eval()
    print(f"[INFO] Model loaded from {model_path}")

    # Connect to TORCS
    print(f"[INFO] Connecting to TORCS ({host}:{port})...")
    client_torcs = snakeoil3.Client(H=host, p=port, vision=False)
    client_torcs.get_servers_input()
    print("[INFO] Connected to TORCS. Starting data collection...\n")

    # Output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).resolve().parents[2] / "data" / f"attempt_model_{timestamp}.csv"
    output_file.parent.mkdir(exist_ok=True)

    # CSV columns: mirror the original attempt's data format
    csv_columns = [
        "timestamp", "distFromStart", "distRaced", "curLapTime", "angle", "speed",
        "trackPos", "track_0", "track_1", "track_2", "track_3", "track_4", "track_5",
        "track_6", "track_7", "track_8", "track_9", "track_10", "track_11", "track_12",
        "track_13", "track_14", "track_15", "track_16", "track_17", "track_18",
        "rpm", "gear", "damage", "wheel_0", "wheel_1", "wheel_2", "wheel_3",
        "steer", "accel", "brake"
    ]

    rows = []
    lap_count = 0
    tempo_giro_precedente = 0.0
    contatore_passi = 0
    marcia_attuale = 1
    sterzata_precedente = 0.0

    print("Running 5 laps. Press Ctrl+C to stop.\n")

    try:
        while lap_count < 5:
            client_torcs.get_servers_input()
            sensori = client_torcs.S.d

            # Lap detection
            tempo_giro_corrente = sensori.get("curLapTime", 0.0)
            if tempo_giro_corrente < tempo_giro_precedente - 1.0:
                lap_count += 1
                print(f"[LAP {lap_count}/5] Completed. Time: {tempo_giro_precedente:.1f}s")
                if lap_count >= 5:
                    break
            tempo_giro_precedente = tempo_giro_corrente

            # Model inference
            vettore_input = build_input_vector(sensori, colonne_input)
            input_normalizzato = (vettore_input - media_normalizzazione) / deviazione_standard
            tensor_input = torch.from_numpy(input_normalizzato).unsqueeze(0).to(dispositivo)

            with torch.no_grad():
                predizione_sterzo, predizione_pedali, logits_marcia = modello_guida(tensor_input)

            angolo_sterzo = float(predizione_sterzo.item()) * 1.8
            acceleratore = float(predizione_pedali[0, 0].item())
            freno = float(predizione_pedali[0, 1].item())
            marcia_predetta = int(logits_marcia.argmax(dim=1).item()) - offset_marce

            # Startup phase
            if contatore_passi < 80:
                acceleratore = 1.0
                freno = 0.0
                angolo_sterzo = angolo_sterzo * 0.5
                if sensori.get("speedX", 0) < 5:
                    marcia_attuale = 1
                elif sensori.get("speedX", 0) < 15:
                    marcia_attuale = 2
                else:
                    marcia_attuale = 3
            else:
                marcia_attuale = marcia_predetta

            # Send action
            client_torcs.R.d["steer"] = angolo_sterzo
            client_torcs.R.d["accel"] = acceleratore
            client_torcs.R.d["brake"] = freno
            client_torcs.R.d["gear"] = marcia_attuale
            client_torcs.respond_to_server()

            # Record data
            track_list = sensori.get("track", [200.0] * 19)
            wheel_list = sensori.get("wheelSpinVel", [0.0] * 4)

            row = {
                "timestamp": time.time(),
                "distFromStart": sensori.get("distFromStart", 0.0),
                "distRaced": sensori.get("distRaced", 0.0),
                "curLapTime": sensori.get("curLapTime", 0.0),
                "angle": sensori.get("angle", 0.0),
                "speed": sensori.get("speedX", 0.0),
                "trackPos": sensori.get("trackPos", 0.0),
            }
            for i in range(19):
                row[f"track_{i}"] = track_list[i] if i < len(track_list) else 200.0
            row["rpm"] = sensori.get("rpm", 0.0)
            row["gear"] = sensori.get("gear", 0)
            row["damage"] = sensori.get("damage", 0.0)
            for i in range(4):
                row[f"wheel_{i}"] = wheel_list[i] if i < len(wheel_list) else 0.0
            row["steer"] = angolo_sterzo
            row["accel"] = acceleratore
            row["brake"] = freno

            rows.append(row)
            contatore_passi += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # Save CSV
    print(f"\n[INFO] Saving {len(rows)} samples to {output_file}")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Data saved: {output_file}")
    print(f"[INFO] Collected {lap_count} complete lap(s), {len(rows)} samples")

    client_torcs.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="TORCS server host")
    parser.add_argument("--port", type=int, default=3001, help="TORCS server port")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
