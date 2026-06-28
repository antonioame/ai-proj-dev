"""Train BC model on keyboard recording."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.behavioral_cloning.dataset import TORCSDataset
from training.behavioral_cloning.model import BCPolicy


def train_bc(csv_path: str, output_path: str = "models/bc_v1.pth", epochs: int = 20, batch_size: int = 32):
    """Train BC model on TORCS telemetry.

    Parameters
    ----------
    csv_path : str
        Path to telemetry CSV
    output_path : str
        Where to save the trained model
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    """
    print(f"Loading data from {csv_path}...")
    dataset = TORCSDataset(csv_path)
    print(f"  Loaded {len(dataset)} samples")
    print(f"  Sensor mean: {dataset.X_mean}")
    print(f"  Sensor std:  {dataset.X_std}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = BCPolicy(input_dim=26, hidden_dims=[128, 64]).to(device)
    print(f"Model: {model}")

    # Loss & optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    print(f"\nTraining for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nModel saved to {output_path}")

    # Also save normalization stats
    stats_path = output_path.with_suffix(".npz")
    import numpy as np
    np.savez(stats_path, mean=dataset.X_mean, std=dataset.X_std)
    print(f"Normalization stats saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC model on TORCS telemetry")
    parser.add_argument("--csv", type=str, default="data/keyboard_20260628_181914.csv", help="Path to telemetry CSV")
    parser.add_argument("--output", type=str, default="models/bc_v1.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_bc(args.csv, args.output, args.epochs, args.batch_size)
