"""Addestra un nuovo checkpoint BC (bc_dagger_v1) unendo il dataset originale
di telemetria (data/driver_*.csv, driver rule-based) con il dataset DAgger
filtrato (scripts/filter_dagger_dataset.py), che usa lo stesso RuleBasedDriver
come oracolo durante il rollout del BC driver reale.

Diversamente dal driver in produzione dell'epoca (il blend di due reti
rettilineo/curva, poi sostituito da bc_tita_v20 il 2026-07-15), qui si
addestra UN SOLO modello unificato sull'unione dei due dataset —
l'architettura BCPolicy (26→128→64, 4 teste) è la stessa.

Non tocca né sovrascrive alcun file esistente: nuovo script, nuovo checkpoint
in _DRIVER/models/bc_dagger_v1.{pth,npz}.

Nota sui dati originali: le sessioni data/driver_*.csv non hanno tutte le
colonne speedY/speedZ (aggiunte in una revisione successiva del formato
sensori) — dove mancano vengono azzerate. Per queste righe la rete vede quindi
un segnale costante (zero) su quei due canali, un'approssimazione dichiarata,
non i valori reali.

Usage:
    python scripts/train_bc_dagger.py --dagger data/dagger_bc_filtered.csv --output-name bc_dagger_v1
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drivers.bc_common import BCPolicy

INPUT_COLS = (
    ["angle", "speed", "speedY", "speedZ", "trackPos"]
    + [f"track_{i}" for i in range(19)]
    + ["rpm", "gear"]
)
FEAT_COLS = [f"feat_{i}" for i in range(26)]


def load_original(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No original telemetry CSVs matched: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "speedY" not in df.columns:
            df["speedY"] = 0.0
        if "speedZ" not in df.columns:
            df["speedZ"] = 0.0
        dfs.append(df)
        print(f"[INFO] Loaded original {f}: {len(df)} raw rows")

    df = pd.concat(dfs, ignore_index=True)
    n_raw = len(df)

    df = df[df["trackPos"].abs() < 0.95]
    if "damage" in df.columns:
        df = df[df["damage"] == 0.0]
    df = df[df["speed"].abs() > 1.0]
    print(f"[INFO] Original dataset after cleaning: {len(df)} / {n_raw} rows")

    X = df[INPUT_COLS].values.astype(np.float32)
    Y = df[["steer", "accel", "brake"]].values.astype(np.float32)
    return X, Y


def load_dagger(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded DAgger dataset {path}: {len(df)} rows")
    X = df[FEAT_COLS].values.astype(np.float32)
    Y = df[["oracle_steer", "oracle_accel", "oracle_brake"]].values.astype(np.float32)
    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="data/driver_*.csv")
    parser.add_argument("--dagger", required=True)
    parser.add_argument("--output-name", default="bc_dagger_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Riproducibilità: senza questi seed l'init dei pesi e lo shuffle rendono ogni run diverso.
    torch.manual_seed(42)
    np.random.seed(42)

    X_orig, Y_orig = load_original(args.original)
    X_dag, Y_dag = load_dagger(args.dagger)

    X = np.concatenate([X_orig, X_dag], axis=0)
    Y = np.concatenate([Y_orig, Y_dag], axis=0)
    print(f"[INFO] Combined dataset: {len(X)} rows ({len(X_orig)} original + {len(X_dag)} DAgger)")

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std

    X_tensor = torch.from_numpy(X_norm.astype(np.float32))
    Y_tensor = torch.from_numpy(Y.astype(np.float32))

    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # NOTA (audit 2026-07-17): split casuale su telemetria sequenziale a 50 Hz —
    # frame adiacenti quasi identici finiscono uno in train e uno in val, quindi la
    # val_loss è ottimistica (leakage temporale). Uno split per giro/sessione sarebbe
    # più rigoroso; la validazione decisiva resta comunque quella in pista.
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = (
                criterion(outputs["steer"].squeeze(), Y_batch[:, 0])
                + criterion(outputs["accel"].squeeze(), Y_batch[:, 1])
                + criterion(outputs["brake"].squeeze(), Y_batch[:, 2])
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = (
                    criterion(outputs["steer"].squeeze(), Y_batch[:, 0])
                    + criterion(outputs["accel"].squeeze(), Y_batch[:, 1])
                    + criterion(outputs["brake"].squeeze(), Y_batch[:, 2])
                )
                val_loss += loss.item()
        val_loss /= len(val_loader)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " <- BEST"
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{marker}")

    output_dir = Path(__file__).resolve().parent.parent / "_DRIVER" / "models"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / f"{args.output_name}.pth"
    stats_path = output_dir / f"{args.output_name}.npz"

    if model_path.exists() or stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {model_path}")

    torch.save(model.state_dict(), model_path)
    np.savez(stats_path, mean=X_mean, std=X_std, input_cols=INPUT_COLS)
    print(f"\n[OK] Model saved to {model_path}")
    print(f"[OK] Stats saved to {stats_path}")
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
