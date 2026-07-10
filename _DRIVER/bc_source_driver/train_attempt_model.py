"""
Addestra il precedente tentativo di driving-net usando un CSV di telemetria esistente.
Usage:
    conda run -n ai_env python _DRIVER/bc_source_driver/train_attempt_model.py --csv data/rule_based_20260628_203648.csv
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse

# Colonne delle feature di input (corrisponde al formato CSV reale della nostra telemetria)
TRACK_COLS = [f"track_{i}" for i in range(19)]
WHEEL_COLS = [f"wheel_{i}" for i in range(4)]
INPUT_COLS = (
    TRACK_COLS +
    ["speed", "trackPos", "angle", "rpm"] +
    WHEEL_COLS
)
GEAR_OFFSET = 1


class DrivingNet(nn.Module):
    """Multi-Layer Perceptron per il controllo di guida."""
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


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Carica e pulisce il CSV di telemetria."""
    df = pd.read_csv(csv_path)
    print(f"[PREPROCESSING] Raw rows: {len(df)}")

    # Scarta le righe con valori mancanti nelle colonne critiche
    df.dropna(subset=INPUT_COLS + ['steer', 'accel', 'brake', 'gear'], inplace=True)

    # Filtra per tenere solo la guida pulita (in pista)
    df = df[df['trackPos'].abs() < 0.9]
    print(f"[PREPROCESSING] Rows after cleaning: {len(df)}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to telemetry CSV")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "attempt_model"),
                        help="Output directory for model")
    args = parser.parse_args()

    # Carica e preprocessa
    df = load_and_clean(args.csv)
    feature_matrix = df[INPUT_COLS].values.astype(np.float32)
    steer_target = df[['steer']].values.astype(np.float32)
    pedals_target = df[['accel', 'brake']].values.astype(np.float32)
    gear_target = (df['gear'].values + GEAR_OFFSET).astype(np.int64)

    # Normalizza
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-6
    normalized_features = (feature_matrix - feature_mean) / feature_std

    # Salva lo scaler
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scaler_info = {
        "mean": feature_mean,
        "std": feature_std,
        "input_cols": INPUT_COLS,
        "gear_offset": GEAR_OFFSET
    }
    scaler_path = output_path / "driving_scaler.pkl"
    joblib.dump(scaler_info, scaler_path)
    print(f"[INFO] Scaler saved: {scaler_path}")

    # Crea il dataset
    full_dataset = TensorDataset(
        torch.from_numpy(normalized_features),
        torch.from_numpy(steer_target),
        torch.from_numpy(pedals_target),
        torch.from_numpy(gear_target)
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Addestramento
    device = torch.device("cpu")
    driving_model = DrivingNet(input_dim=len(INPUT_COLS)).to(device)
    optimizer = torch.optim.Adam(driving_model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    total_epochs = 60

    print("[INFO] Training started...")
    for epoch in range(1, total_epochs + 1):
        driving_model.train()
        epoch_total_loss = 0.0
        for batch_x, batch_y_steer, batch_y_pedals, batch_y_gear in train_loader:
            batch_x, batch_y_steer, batch_y_pedals, batch_y_gear = (
                batch_x.to(device), batch_y_steer.to(device),
                batch_y_pedals.to(device), batch_y_gear.to(device)
            )
            pred_steer, pred_pedals, pred_gear = driving_model(batch_x)
            loss_value = (
                2.0 * mse_loss_fn(pred_steer, batch_y_steer) +
                1.0 * mse_loss_fn(pred_pedals, batch_y_pedals) +
                0.3 * ce_loss_fn(pred_gear, batch_y_gear)
            )
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            epoch_total_loss += loss_value.item() * batch_x.size(0)

        train_loss = epoch_total_loss / len(train_dataset)

        driving_model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_steer, batch_y_pedals, batch_y_gear in val_loader:
                batch_x, batch_y_steer, batch_y_pedals, batch_y_gear = (
                    batch_x.to(device), batch_y_steer.to(device),
                    batch_y_pedals.to(device), batch_y_gear.to(device)
                )
                pred_steer, pred_pedals, pred_gear = driving_model(batch_x)
                loss_value = (
                    2.0 * mse_loss_fn(pred_steer, batch_y_steer) +
                    1.0 * mse_loss_fn(pred_pedals, batch_y_pedals) +
                    0.3 * ce_loss_fn(pred_gear, batch_y_gear)
                )
                val_total_loss += loss_value.item() * batch_x.size(0)

        val_loss = val_total_loss / len(val_dataset)
        lr_scheduler.step()

        save_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = output_path / "driving_model.pt"
            torch.save(driving_model.state_dict(), model_path)
            save_marker = " <- [SAVED]"

        print(f"Epoch {epoch:3d}/{total_epochs} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}{save_marker}")

    print(f"\n[OK] Training complete. Model saved to {output_path / 'driving_model.pt'}")


if __name__ == "__main__":
    main()
