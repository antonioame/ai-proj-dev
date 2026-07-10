"""
Driver sorgente usato per generare la telemetria con cui è stato addestrato
_DRIVER/models/bc_from_attempt1_v1. È un precedente tentativo di driving-net,
mantenuto solo per rigenerare campioni di training.
Rieseguilo se servono campioni nuovi per riaddestrare quel modello BC.

Usage:
    conda run -n ai_env python _DRIVER/bc_source_driver/run_attempt_model.py [--host localhost] [--port 3001]

Il modello deve essere già addestrato:
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
    """Precedente tentativo di driving-net (precursore del modello BC rettilineo)."""
    def __init__(self, input_dim: int, num_gears: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head_steer = nn.Linear(64, 1)
        self.head_accel_brake = nn.Linear(64, 2)
        self.head_gear = nn.Linear(64, num_gears)

    def forward(self, input_data):
        hidden_layer = self.backbone(input_data)
        return (
            torch.tanh(self.head_steer(hidden_layer)),
            torch.sigmoid(self.head_accel_brake(hidden_layer)),
            self.head_gear(hidden_layer),
        )


def build_input_vector(sensor_state, selected_columns):
    """Costruisce il vettore di input grezzo dai sensori (formato TORCS); la normalizzazione avviene a parte, lato chiamante."""
    track_distances = sensor_state.get("track", [200.0] * 19)
    wheel_rotation = sensor_state.get("wheelSpinVel", [0.0] * 4)

    if len(track_distances) != 19:
        track_distances = [200.0] * 19
    if len(wheel_rotation) != 4:
        wheel_rotation = [0.0] * 4

    feature_dict = {}
    for i in range(19):
        feature_dict[f"track_{i}"] = float(track_distances[i])
    for i in range(4):
        feature_dict[f"wheel_{i}"] = float(wheel_rotation[i])

    # Mappa speedX di TORCS su 'speed' (come nei nostri dati CSV di training)
    feature_dict["speed"] = float(sensor_state.get("speedX", 0))
    feature_dict["trackPos"] = float(sensor_state.get("trackPos", 0))
    feature_dict["angle"] = float(sensor_state.get("angle", 0))
    feature_dict["rpm"] = float(sensor_state.get("rpm", 0))

    return np.array([feature_dict[col] for col in selected_columns], dtype=np.float32)


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

    # Carica lo scaler
    print("[INFO] Loading model and scaler...")
    scaler_data = joblib.load(scaler_path)
    norm_mean = scaler_data["mean"]
    norm_std = scaler_data["std"]
    input_cols = scaler_data["input_cols"]
    gear_offset = scaler_data["gear_offset"]

    device = torch.device("cpu")
    driving_model = DrivingNet(input_dim=len(input_cols)).to(device)
    driving_model.load_state_dict(torch.load(model_path, map_location=device))
    driving_model.eval()
    print(f"[INFO] Model loaded from {model_path}")

    # Connessione a TORCS
    print(f"[INFO] Connecting to TORCS ({host}:{port})...")
    torcs_client = snakeoil3.Client(H=host, p=port, vision=False)
    torcs_client.get_servers_input()
    print("[INFO] Connected to TORCS. Starting data collection...\n")

    # CSV di output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).resolve().parents[2] / "data" / f"attempt_model_{timestamp}.csv"
    output_file.parent.mkdir(exist_ok=True)

    # Colonne CSV: rispecchiano il formato dati dell'attempt originale
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
    prev_lap_time = 0.0
    step_counter = 0
    current_gear = 1
    prev_steer = 0.0

    print("Running 5 laps. Press Ctrl+C to stop.\n")

    try:
        while lap_count < 5:
            torcs_client.get_servers_input()
            sensors = torcs_client.S.d

            # Rilevamento giro
            cur_lap_time = sensors.get("curLapTime", 0.0)
            if cur_lap_time < prev_lap_time - 1.0:
                lap_count += 1
                print(f"[LAP {lap_count}/5] Completed. Time: {prev_lap_time:.1f}s")
                if lap_count >= 5:
                    break
            prev_lap_time = cur_lap_time

            # Inferenza del modello
            input_vector = build_input_vector(sensors, input_cols)
            normalized_input = (input_vector - norm_mean) / norm_std
            input_tensor = torch.from_numpy(normalized_input).unsqueeze(0).to(device)

            with torch.no_grad():
                steer_pred, pedals_pred, gear_logits = driving_model(input_tensor)

            steer_angle = float(steer_pred.item()) * 1.8
            accel = float(pedals_pred[0, 0].item())
            brake = float(pedals_pred[0, 1].item())
            predicted_gear = int(gear_logits.argmax(dim=1).item()) - gear_offset

            # Fase di avvio
            if step_counter < 80:
                accel = 1.0
                brake = 0.0
                steer_angle = steer_angle * 0.5
                if sensors.get("speedX", 0) < 5:
                    current_gear = 1
                elif sensors.get("speedX", 0) < 15:
                    current_gear = 2
                else:
                    current_gear = 3
            else:
                current_gear = predicted_gear

            # Invia l'azione
            torcs_client.R.d["steer"] = steer_angle
            torcs_client.R.d["accel"] = accel
            torcs_client.R.d["brake"] = brake
            torcs_client.R.d["gear"] = current_gear
            torcs_client.respond_to_server()

            # Registra i dati
            track_list = sensors.get("track", [200.0] * 19)
            wheel_list = sensors.get("wheelSpinVel", [0.0] * 4)

            row = {
                "timestamp": time.time(),
                "distFromStart": sensors.get("distFromStart", 0.0),
                "distRaced": sensors.get("distRaced", 0.0),
                "curLapTime": sensors.get("curLapTime", 0.0),
                "angle": sensors.get("angle", 0.0),
                "speed": sensors.get("speedX", 0.0),
                "trackPos": sensors.get("trackPos", 0.0),
            }
            for i in range(19):
                row[f"track_{i}"] = track_list[i] if i < len(track_list) else 200.0
            row["rpm"] = sensors.get("rpm", 0.0)
            row["gear"] = sensors.get("gear", 0)
            row["damage"] = sensors.get("damage", 0.0)
            for i in range(4):
                row[f"wheel_{i}"] = wheel_list[i] if i < len(wheel_list) else 0.0
            row["steer"] = steer_angle
            row["accel"] = accel
            row["brake"] = brake

            rows.append(row)
            step_counter += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # Salva il CSV
    print(f"\n[INFO] Saving {len(rows)} samples to {output_file}")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Data saved: {output_file}")
    print(f"[INFO] Collected {lap_count} complete lap(s), {len(rows)} samples")

    torcs_client.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="TORCS server host")
    parser.add_argument("--port", type=int, default=3001, help="TORCS server port")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
