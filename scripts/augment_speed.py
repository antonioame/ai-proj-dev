"""
Aumenta i dati di telemetria per una guida più aggressiva.

Vengono modificate solo le colonne di OUTPUT (steer, accel, brake).
Le colonne di INPUT (speed, track_*, angle, ecc.) restano intatte per
evitare un disallineamento tra la distribuzione di training e quella di
inferenza.

Strategia:
  - accel:  +7%  (più gas → rettilinei e uscite di curva più veloci)
  - steer:  +5%  (linee di curva più strette)
  - brake:  -10% (frenata più tardiva → più velocità in ingresso curva)
  Tutto limitato agli intervalli validi.

Usage:
    conda run -n ai_env python scripts/augment_speed.py \
        --input data/old_driver_*.csv \
        --output data/old_driver_augmented_*.csv
"""

import pandas as pd
import argparse


def augment_csv(
    input_path: str,
    output_path: str,
    accel_pct: float = 7.0,
    steer_pct: float = 5.0,
    brake_reduction_pct: float = 10.0,
):
    print(f"[INFO] Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Accel range:  {df['accel'].min():.3f} - {df['accel'].max():.3f}")
    print(f"[INFO] Steer range:  {df['steer'].min():.3f} - {df['steer'].max():.3f}")
    print(f"[INFO] Brake range:  {df['brake'].min():.3f} - {df['brake'].max():.3f}")

    # Tocca solo le colonne di OUTPUT — gli input dei sensori restano intatti
    df["accel"] = (df["accel"] * (1.0 + accel_pct / 100.0)).clip(0.0, 1.0)
    df["steer"] = (df["steer"] * (1.0 + steer_pct / 100.0)).clip(-1.0, 1.0)
    df["brake"] = (df["brake"] * (1.0 - brake_reduction_pct / 100.0)).clip(0.0, 1.0)

    print("\n[AUGMENTATION] Applied (outputs only, sensor inputs unchanged):")
    print(f"  - accel: +{accel_pct}%  → {df['accel'].min():.3f} - {df['accel'].max():.3f}")
    print(f"  - steer: +{steer_pct}%  → {df['steer'].min():.3f} - {df['steer'].max():.3f}")
    print(f"  - brake: -{brake_reduction_pct}%  → {df['brake'].min():.3f} - {df['brake'].max():.3f}")

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accel-pct", type=float, default=7.0)
    parser.add_argument("--steer-pct", type=float, default=5.0)
    parser.add_argument("--brake-reduction-pct", type=float, default=10.0)
    args = parser.parse_args()

    augment_csv(
        args.input, args.output,
        accel_pct=args.accel_pct,
        steer_pct=args.steer_pct,
        brake_reduction_pct=args.brake_reduction_pct,
    )


if __name__ == "__main__":
    main()
