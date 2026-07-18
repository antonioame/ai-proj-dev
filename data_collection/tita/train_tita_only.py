"""Addestra un modello BC candidato usando ESCLUSIVAMENTE la telemetria di
tita (data_collection/tita/converted/*.csv), senza unirla al dataset
originale delle sessioni dell'utente (per coerenza: dataset non contaminati
tra loro).

Stessa architettura BCPolicy (26->128->64, 4 teste) e stesso schema di
salvataggio (.pth + .npz con mean/std/input_cols) di train_bc_dagger.py, cosi'
il candidato resta caricabile con lo stesso _load_bc_model() di
_DRIVER/driver.py per una valutazione in pista.

Non tocca ne' sovrascrive _DRIVER/, data/, training/, drivers/: il checkpoint
va in data_collection/tita/candidate_models/, fuori da _DRIVER/models/, finche'
non viene promosso esplicitamente dopo valutazione.

Usage:
    python data_collection/tita/train_tita_only.py --output-name bc_tita_v1
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drivers.bc_common import BCPolicy

INPUT_COLS = (
    ["angle", "speed", "speedY", "speedZ", "trackPos"]
    + [f"track_{i}" for i in range(19)]
    + ["rpm", "gear"]
)

CONVERTED_DIR = Path(__file__).parent / "converted"
CANDIDATE_DIR = Path(__file__).parent / "candidate_models"


def load_tita(pattern: str, extra_pattern: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nessun CSV convertito trovato: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"[INFO] Loaded {f}: {len(df)} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Dataset tita combinato: {len(df)} righe (gia' pulito da convert_tita_csv.py)")

    if extra_pattern:
        extra_files = sorted(glob.glob(extra_pattern))
        if not extra_files:
            raise FileNotFoundError(f"Nessun CSV extra trovato: {extra_pattern}")
        extra_dfs = []
        for f in extra_files:
            edf = pd.read_csv(f)
            extra_dfs.append(edf)
            print(f"[INFO] Loaded extra (recovery) {f}: {len(edf)} rows")
        extra_df = pd.concat(extra_dfs, ignore_index=True)
        df = pd.concat([df, extra_df], ignore_index=True)
        print(f"[INFO] Dataset totale con esempi extra: {len(df)} righe")

    X = df[INPUT_COLS].values.astype(np.float32)
    Y = df[["steer", "accel", "brake"]].values.astype(np.float32)
    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(CONVERTED_DIR / "*.csv"))
    parser.add_argument("--extra", default=None, help="Glob opzionale di CSV extra da aggiungere (es. esempi di recupero DAgger-style)")
    parser.add_argument("--output-name", default="bc_tita_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Riproducibilità: senza questi seed l'init dei pesi e lo shuffle rendono ogni run diverso.
    torch.manual_seed(42)
    np.random.seed(42)

    X, Y = load_tita(args.input, args.extra)

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

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CANDIDATE_DIR / f"{args.output_name}.pth"
    stats_path = CANDIDATE_DIR / f"{args.output_name}.npz"

    if model_path.exists() or stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {model_path}")

    torch.save(model.state_dict(), model_path)
    np.savez(stats_path, mean=X_mean, std=X_std, input_cols=INPUT_COLS)
    print(f"\n[OK] Model saved to {model_path}")
    print(f"[OK] Stats saved to {stats_path}")
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
