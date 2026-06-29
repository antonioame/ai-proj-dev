"""
Augment telemetry data by increasing speed and acceleration targets.
Useful for training models to drive more aggressively based on expert demonstrations.

Usage:
    conda run -n ai_env python scripts/augment_speed.py --input data/friend_model_*.csv --output data/friend_model_augmented_*.csv

Strategy:
  - Speed: +5% (teaches the model to maintain higher velocity)
  - Accel: +3% (more aggressive throttle but conservative on braking for safety)
  - Brake: unchanged (preserve safety margins)
"""

import pandas as pd
import argparse
from pathlib import Path


def augment_csv(input_path: str, output_path: str, speed_pct: float = 5.0, accel_pct: float = 3.0):
    """
    Augment telemetry CSV by increasing speed and accel targets.

    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        speed_pct: Percentage increase for speed column (e.g., 5.0 = +5%)
        accel_pct: Percentage increase for accel column (e.g., 3.0 = +3%)
    """
    print(f"[INFO] Loading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"[INFO] Original shape: {df.shape}")
    print(f"[INFO] Speed range: {df['speed'].min():.1f} - {df['speed'].max():.1f} km/h")
    print(f"[INFO] Accel range: {df['accel'].min():.3f} - {df['accel'].max():.3f}")

    # Increase speed target
    df['speed'] = df['speed'] * (1.0 + speed_pct / 100.0)

    # Increase accel target (clip to [0, 1] range)
    df['accel'] = (df['accel'] * (1.0 + accel_pct / 100.0)).clip(0, 1.0)

    # Brake unchanged (safety)

    print(f"\n[AUGMENTATION] Applied:")
    print(f"  - Speed: +{speed_pct}%")
    print(f"  - Accel: +{accel_pct}% (clipped to [0, 1])")
    print(f"  - Brake: unchanged")

    print(f"\n[INFO] New speed range: {df['speed'].min():.1f} - {df['speed'].max():.1f} km/h")
    print(f"[INFO] New accel range: {df['accel'].min():.3f} - {df['accel'].max():.3f}")

    print(f"\n[INFO] Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"[OK] Augmented data saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Augment telemetry data for performance")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--speed-pct", type=float, default=5.0, help="Speed increase percentage (default 5)")
    parser.add_argument("--accel-pct", type=float, default=3.0, help="Accel increase percentage (default 3)")
    args = parser.parse_args()

    augment_csv(args.input, args.output, speed_pct=args.speed_pct, accel_pct=args.accel_pct)


if __name__ == "__main__":
    main()
