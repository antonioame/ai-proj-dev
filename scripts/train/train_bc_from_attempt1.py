"""
Addestra il modello di Behavioral Cloning sui dati del precedente tentativo
di driving-net.

Usage:
    # Primo: raccogliere 5 giri dall'attempt model
    conda run -n ai_env python _DRIVER/bc_source_driver/train_attempt_model.py --csv data/rule_based_20260628_203648.csv
    conda run -n ai_env python _DRIVER/bc_source_driver/run_attempt_model.py

    # Secondo: aumentare i dati per una guida più aggressiva
    # (nota: augment_speed.py non esiste più, vedi scripts/train/prepare_training_data.py)
    conda run -n ai_env python scripts/augment_speed.py \\
        --input data/attempt_model_20260629_*.csv \\
        --output data/attempt_model_augmented_20260629_*.csv

    # Terzo: addestrare il modello BC su entrambi i dataset
    conda run -n ai_env python scripts/train/train_bc_from_attempt1.py \\
        --original data/attempt_model_20260629_*.csv \\
        --augmented data/attempt_model_augmented_20260629_*.csv \\
        --output-name bc_from_attempt1_v2
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import glob

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from drivers.bc_common import BCPolicy


def load_csv_files(pattern: str):
    """Carica e concatena i file CSV che corrispondono al pattern."""
    files = glob.glob(pattern)
    if not files:
        print(f"[ERROR] No CSV files found matching: {pattern}")
        sys.exit(1)

    dfs = []
    for f in files:
        print(f"[INFO] Loading {f}")
        dfs.append(pd.read_csv(f))

    return pd.concat(dfs, ignore_index=True)


def build_bc_dataset(csv_path: str):
    """Carica il CSV e costruisce il dataset di training per BC."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} samples from {csv_path}")

    # Feature di input: angle, speed, speedY, speedZ, trackPos, track_0-18, rpm, gear
    input_cols = (
        ["angle", "speed", "speedY", "speedZ", "trackPos"] +
        [f"track_{i}" for i in range(19)] +
        ["rpm", "gear"]
    )

    # Target di output
    output_cols = ["steer", "accel", "brake"]

    # Filtra: solo campioni in pista
    df_clean = df[df["trackPos"].abs() < 0.95].copy()
    print(f"[INFO] After filtering (trackPos < 0.95): {len(df_clean)} samples")

    X = df_clean[input_cols].values.astype(np.float32)
    Y = df_clean[output_cols].values.astype(np.float32)

    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std

    return torch.from_numpy(X_norm), torch.from_numpy(Y), X_mean, X_std, input_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True,
                        help="Original CSV pattern (e.g., data/attempt_model_*.csv)")
    parser.add_argument("--augmented", type=str, required=True,
                        help="Augmented CSV pattern (e.g., data/attempt_model_augmented_*.csv)")
    parser.add_argument("--output-name", type=str, default="bc_from_attempt1_v2",
                        help="Output model name (saved as _DRIVER/models/<name>.pth, _DRIVER/models/<name>.npz)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Riproducibilità: senza questi seed l'init dei pesi e lo shuffle rendono ogni run diverso.
    torch.manual_seed(42)
    np.random.seed(42)

    print("[INFO] Loading datasets...")
    df_original = load_csv_files(args.original)
    df_augmented = load_csv_files(args.augmented)

    # 80% campionato dall'originale + 20% dall'aumentato (frazioni di ciascun
    # dataset separatamente, non della miscela finale).
    df_combined = pd.concat([
        df_original.sample(frac=0.8, random_state=42),
        df_augmented.sample(frac=0.2, random_state=42),
    ], ignore_index=True)
    print(f"[INFO] Combined dataset: {len(df_combined)} samples (80% of original + 20% of augmented)")

    # Costruisce il dataset
    input_cols = (
        ["angle", "speed", "speedY", "speedZ", "trackPos"] +
        [f"track_{i}" for i in range(19)] +
        ["rpm", "gear"]
    )
    output_cols = ["steer", "accel", "brake"]

    df_clean = df_combined[df_combined["trackPos"].abs() < 0.95].copy()
    print(f"[INFO] After filtering: {len(df_clean)} samples")

    X = df_clean[input_cols].values.astype(np.float32)
    Y = df_clean[output_cols].values.astype(np.float32)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std

    X_tensor = torch.from_numpy(X_norm)
    Y_tensor = torch.from_numpy(Y)

    print(f"\n[INFO] Input shape: {X_tensor.shape}")
    print(f"[INFO] Output shape: {Y_tensor.shape}")

    # Split train/val
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # NOTA : split casuale su telemetria sequenziale a 50 Hz:
    # frame adiacenti quasi identici finiscono uno in train e uno in val, quindi la
    # val_loss è ottimistica (leakage temporale). Uno split per giro/sessione sarebbe
    # più rigoroso; la validazione decisiva resta comunque quella in pista.
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Training on device: {device}")

    model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    print("[INFO] Starting training...\n")
    for epoch in range(1, args.epochs + 1):
        # Addestramento
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Loss: steer + accel + brake (peso uguale)
            steer_loss = criterion(outputs["steer"].squeeze(), Y_batch[:, 0])
            accel_loss = criterion(outputs["accel"].squeeze(), Y_batch[:, 1])
            brake_loss = criterion(outputs["brake"].squeeze(), Y_batch[:, 2])
            loss = steer_loss + accel_loss + brake_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)

                steer_loss = criterion(outputs["steer"].squeeze(), Y_batch[:, 0])
                accel_loss = criterion(outputs["accel"].squeeze(), Y_batch[:, 1])
                brake_loss = criterion(outputs["brake"].squeeze(), Y_batch[:, 2])
                loss = steer_loss + accel_loss + brake_loss

                val_loss += loss.item()

        val_loss /= len(val_loader)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " <- BEST"

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{marker}")

    # Salva il modello
    output_dir = Path(__file__).resolve().parent.parent.parent / "_DRIVER" / "models"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / f"{args.output_name}.pth"
    stats_path = output_dir / f"{args.output_name}.npz"

    if model_path.exists() or stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {model_path}")

    torch.save(model.state_dict(), model_path)
    np.savez(stats_path, mean=X_mean, std=X_std, input_cols=input_cols)

    print(f"\n[OK] Model saved to {model_path}")
    print(f"[OK] Stats saved to {stats_path}")
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")

    # Mostra le istruzioni
    print("\n[NEXT] To use this model:")
    print(f"  1. Update _DRIVER/driver.py to load: _DRIVER/models/{args.output_name}.pth")
    print("  2. Run: conda run -n ai_env python scripts/run/run_agent.py --laps 1")


if __name__ == "__main__":
    main()
