"""
Train Behavioral Cloning model from friend's driving data.

Usage:
    # First: collect 5 laps from friend's model
    conda run -n ai_env python scripts/train_friend_model.py --csv data/rule_based_20260628_203648.csv
    conda run -n ai_env python scripts/run_friend_model.py

    # Second: augment data for more aggressive driving
    conda run -n ai_env python scripts/augment_speed.py \\
        --input data/friend_model_20260629_*.csv \\
        --output data/friend_model_augmented_20260629_*.csv

    # Third: train BC model on both datasets
    conda run -n ai_env python scripts/train_bc_from_friend.py \\
        --original data/friend_model_20260629_*.csv \\
        --augmented data/friend_model_augmented_20260629_*.csv \\
        --output-name bc_from_friend_v1
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


class BCPolicy(nn.Module):
    """Behavioral Cloning MLP for TORCS driving."""
    def __init__(self, input_dim: int = 26, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.head_steer = nn.Linear(prev_dim, 1)  # tanh(steering angle)
        self.head_accel = nn.Linear(prev_dim, 1)  # sigmoid(acceleration)
        self.head_brake = nn.Linear(prev_dim, 1)  # sigmoid(brake)
        self.head_gear = nn.Linear(prev_dim, 1)   # gear selection (regression)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "steer": torch.tanh(self.head_steer(features)),
            "accel": torch.sigmoid(self.head_accel(features)),
            "brake": torch.sigmoid(self.head_brake(features)),
            "gear": self.head_gear(features),
        }


def load_csv_files(pattern: str):
    """Load and concatenate CSV files matching pattern."""
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
    """Load CSV and build training dataset for BC."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} samples from {csv_path}")

    # Input features: angle, speed, speedY, speedZ, trackPos, track_0-18, rpm, gear
    input_cols = (
        ["angle", "speed", "speedY", "speedZ", "trackPos"] +
        [f"track_{i}" for i in range(19)] +
        ["rpm", "gear"]
    )

    # Output targets
    output_cols = ["steer", "accel", "brake"]

    # Filter: only on-track samples
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
                        help="Original CSV pattern (e.g., data/friend_model_*.csv)")
    parser.add_argument("--augmented", type=str, required=True,
                        help="Augmented CSV pattern (e.g., data/friend_model_augmented_*.csv)")
    parser.add_argument("--output-name", type=str, default="bc_from_friend_v1",
                        help="Output model name (saved as models/<name>.pth, models/<name>.npz)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("[INFO] Loading datasets...")
    df_original = load_csv_files(args.original)
    df_augmented = load_csv_files(args.augmented)

    # Combine datasets (80% original, 20% augmented for gentle augmentation)
    df_combined = pd.concat([
        df_original.sample(frac=0.8, random_state=42),
        df_augmented.sample(frac=0.2, random_state=42),
    ], ignore_index=True)
    print(f"[INFO] Combined dataset: {len(df_combined)} samples (80% original, 20% augmented)")

    # Build dataset
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

    # Train/val split
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Training on device: {device}")

    model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    print("[INFO] Starting training...\n")
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Loss: steer + accel + brake (equal weight)
            steer_loss = criterion(outputs["steer"].squeeze(), Y_batch[:, 0])
            accel_loss = criterion(outputs["accel"].squeeze(), Y_batch[:, 1])
            brake_loss = criterion(outputs["brake"].squeeze(), Y_batch[:, 2])
            loss = steer_loss + accel_loss + brake_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
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

    # Save model
    output_dir = Path(__file__).resolve().parent.parent / "models"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / f"{args.output_name}.pth"
    stats_path = output_dir / f"{args.output_name}.npz"

    torch.save(model.state_dict(), model_path)
    np.savez(stats_path, mean=X_mean, std=X_std, input_cols=input_cols)

    print(f"\n[OK] Model saved to {model_path}")
    print(f"[OK] Stats saved to {stats_path}")
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")

    # Show instructions
    print(f"\n[NEXT] To use this model:")
    print(f"  1. Update drivers/bc/driver.py to load: models/{args.output_name}.pth")
    print(f"  2. Run: conda run -n ai_env python scripts/run_agent.py --driver bc --laps 1")


if __name__ == "__main__":
    main()
