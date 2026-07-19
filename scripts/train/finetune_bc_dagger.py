"""Fine-tuning a caldo (warm-start) di bc_from_olddriver_v1 usando SOLO il
dataset DAgger filtrato (data/dagger_bc_filtered.csv) come dati di
correzione — non un retraining da zero.

Diversamente da scripts/train/train_bc_dagger.py (retraining completo su dataset
originale + DAgger, invalidato: il dataset "originale" disponibile nel repo
non è quello che ha prodotto bc_from_olddriver_v1), qui si parte dai pesi
ESISTENTI di bc_from_olddriver_v1.pth e si applicano poche epoche a learning
rate basso solo sulle 27.366 righe DAgger, per correggere senza disgregare
il comportamento già buono.

Cruciale: la normalizzazione (mean/std) usata è quella ESISTENTE di
bc_from_olddriver_v1.npz, non ricalcolata sui nuovi dati — i pesi caricati
sono validi solo per quella normalizzazione.

Non tocca né sovrascrive bc_from_olddriver_v1.{pth,npz} né alcun altro file
esistente: nuovo checkpoint in _DRIVER/models/bc_finetune_dagger_v1.{pth,npz}.

Usage:
    python scripts/train/finetune_bc_dagger.py --dagger data/dagger_bc_filtered.csv \
        --epochs 8 --lr 5e-5 --output-name bc_finetune_dagger_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from drivers.bc_common import BCPolicy

FEAT_COLS = [f"feat_{i}" for i in range(26)]
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "_DRIVER" / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-name", default="bc_from_olddriver_v1")
    parser.add_argument("--dagger", default="data/dagger_bc_filtered.csv")
    parser.add_argument("--output-name", default="bc_finetune_dagger_v1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    # Riproducibilità: senza questi seed lo shuffle del DataLoader rende ogni run diverso
    # (l'init dei pesi qui non conta: si parte dai pesi caricati del modello base).
    torch.manual_seed(42)
    np.random.seed(42)

    base_model_path = MODELS_DIR / f"{args.base_name}.pth"
    base_stats_path = MODELS_DIR / f"{args.base_name}.npz"

    stats = np.load(base_stats_path, allow_pickle=True)
    X_mean = stats["mean"].astype(np.float32)
    X_std = stats["std"].astype(np.float32)
    input_cols = stats["input_cols"]
    print(f"[INFO] Loaded normalization stats from {base_stats_path} (unchanged, reused as-is)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    print(f"[INFO] Loaded base weights from {base_model_path} (warm-start)")

    df = pd.read_csv(args.dagger)
    print(f"[INFO] Loaded DAgger dataset {args.dagger}: {len(df)} rows")
    X = df[FEAT_COLS].values.astype(np.float32)
    Y = df[["oracle_steer", "oracle_accel", "oracle_brake"]].values.astype(np.float32)

    # Normalizza con le stats ESISTENTI del modello base, non con nuove stats.
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    def _eval_loss(loader) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = (
                    criterion(outputs["steer"].squeeze(), Y_batch[:, 0])
                    + criterion(outputs["accel"].squeeze(), Y_batch[:, 1])
                    + criterion(outputs["brake"].squeeze(), Y_batch[:, 2])
                )
                total += loss.item()
        return total / len(loader)

    print(f"[INFO] Val loss BEFORE fine-tuning: {_eval_loss(val_loader):.4f}")

    best_val_loss = float("inf")
    best_state = None
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

        val_loss = _eval_loss(val_loader)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " <- BEST"
        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Restored best-val-loss weights (val_loss={best_val_loss:.4f})")

    model_path = MODELS_DIR / f"{args.output_name}.pth"
    stats_path = MODELS_DIR / f"{args.output_name}.npz"
    if model_path.exists() or stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {model_path}")

    torch.save(model.state_dict(), model_path)
    np.savez(stats_path, mean=X_mean, std=X_std, input_cols=input_cols)
    print(f"\n[OK] Model saved to {model_path}")
    print(f"[OK] Stats saved to {stats_path} (identical normalization to {args.base_name})")


if __name__ == "__main__":
    main()
