"""Behavioral Cloning training script.

Usage:
    python -m training.behavioral_cloning.train \\
        --data data/human_20240101_120000.csv \\
        --output models/bc_v1.pth \\
        --epochs 50 --batch-size 256

MPS (Apple Silicon) is used automatically when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import TelemetryDataset
from .model import MLPPolicy


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    data_paths: list[Path],
    output_path: Path,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    hidden_dims: list[int] | None = None,
) -> dict:
    device = _get_device()
    print(f"Training on device: {device}")

    dataset = TelemetryDataset(data_paths)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MLPPolicy(
        input_dim=dataset.input_dim,
        hidden_dims=hidden_dims or [256, 256, 128],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)

            loss = (
                mse(out["steer"], y[:, 0])
                + mse(out["accel"], y[:, 1])
                + mse(out["brake"], y[:, 2])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += (
                    mse(out["steer"], y[:, 0])
                    + mse(out["accel"], y[:, 1])
                    + mse(out["brake"], y[:, 2])
                ).item()

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        history.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss})
        print(f"Epoch {epoch:3d} | train {t_loss:.4f} | val {v_loss:.4f}")

    # Save model + normalisation stats
    sensor_mean, sensor_std = dataset.normalisation
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": dataset.input_dim,
            "output_dim": dataset.output_dim,
            "sensor_mean": sensor_mean,
            "sensor_std": sensor_std,
            "hidden_dims": hidden_dims or [256, 256, 128],
            "history": history,
        },
        output_path,
    )
    print(f"Model saved to {output_path}")
    return history[-1] if history else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral Cloning training")
    parser.add_argument("--data", nargs="+", required=True, help="CSV telemetry file(s)")
    parser.add_argument("--output", default="models/bc_v1.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        data_paths=[Path(p) for p in args.data],
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
