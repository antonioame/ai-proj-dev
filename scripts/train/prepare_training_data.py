"""
Pulisce e aumenta il CSV di telemetria per il training BC.

Due operazioni:
  1. CLEAN: rimuove colonne duplicate, scarta righe fuori pista, scarta righe con danni
  2. AUGMENT: modifica solo le colonne di OUTPUT (steer, accel, brake)
              gli input dei sensori non vengono mai toccati per evitare un
              disallineamento tra training e inferenza

Strategia di augmentation (orientata alla performance):
  - accel:  +7%  limitato a [0, 1]   — più gas in uscita di curva e sui rettilinei
  - steer:  +5%  limitato a [-1, 1]  — linea di curva più stretta
  - brake:  -10% limitato a [0, 1]   — punto di frenata più tardivo

Usage:
    conda run -n ai_env python scripts/train/prepare_training_data.py \
        --input data/old_driver_20260629_170825.csv \
        --output-clean data/old_driver_clean.csv \
        --output-augmented data/old_driver_augmented.csv
"""

import argparse
import pandas as pd

# Colonne di input dei sensori — mai modificate
INPUT_COLS = (
    ["timestamp", "angle", "speed", "speedY", "speedZ", "trackPos"] +
    [f"track_{i}" for i in range(19)] +
    ["rpm", "gear", "damage", "distRaced", "curLapTime"]
)

# Colonne di output azione — aumentate
OUTPUT_COLS = ["steer", "accel", "brake"]

EXPECTED_COLS = INPUT_COLS + OUTPUT_COLS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove duplicati e righe non valide."""
    print(f"[CLEAN] Input rows: {len(df)}")

    # Scarta le colonne duplicate (mantiene la prima occorrenza)
    df = df.loc[:, ~df.columns.duplicated()]

    # Mantiene solo le colonne attese (scarta ogni extra)
    cols_present = [c for c in EXPECTED_COLS if c in df.columns]
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"[CLEAN] WARNING: missing columns: {missing}")
    df = df[cols_present]

    # Scarta i NaN nelle colonne critiche
    critical = ["angle", "speed", "trackPos", "steer", "accel", "brake"] + [f"track_{i}" for i in range(19)]
    df = df.dropna(subset=[c for c in critical if c in df.columns])

    # Mantiene solo i campioni in pista
    df = df[df["trackPos"].abs() < 0.95]

    # Scarta i campioni con danni (l'auto ha urtato qualcosa)
    if "damage" in df.columns:
        df = df[df["damage"] == 0.0]

    # Scarta i campioni di avvio in cui l'auto è ancora ferma
    df = df[df["speed"].abs() > 1.0]

    print(f"[CLEAN] Output rows: {len(df)}")
    return df.reset_index(drop=True)


def augment(df: pd.DataFrame, accel_pct: float, steer_pct: float, brake_reduction_pct: float) -> pd.DataFrame:
    """Aumenta solo gli output azione, lasciando intatti gli input dei sensori."""
    df = df.copy()

    before = {
        "accel": (df["accel"].min(), df["accel"].max()),
        "steer": (df["steer"].min(), df["steer"].max()),
        "brake": (df["brake"].min(), df["brake"].max()),
    }

    df["accel"] = (df["accel"] * (1.0 + accel_pct / 100.0)).clip(0.0, 1.0)
    df["steer"] = (df["steer"] * (1.0 + steer_pct / 100.0)).clip(-1.0, 1.0)
    df["brake"] = (df["brake"] * (1.0 - brake_reduction_pct / 100.0)).clip(0.0, 1.0)

    print("\n[AUGMENT] Sensor inputs: unchanged")
    print(f"[AUGMENT] accel: {before['accel'][0]:.3f}-{before['accel'][1]:.3f}  →  "
          f"{df['accel'].min():.3f}-{df['accel'].max():.3f}  (+{accel_pct}%)")
    print(f"[AUGMENT] steer: {before['steer'][0]:.3f}-{before['steer'][1]:.3f}  →  "
          f"{df['steer'].min():.3f}-{df['steer'].max():.3f}  (+{steer_pct}%)")
    print(f"[AUGMENT] brake: {before['brake'][0]:.3f}-{before['brake'][1]:.3f}  →  "
          f"{df['brake'].min():.3f}-{df['brake'].max():.3f}  (-{brake_reduction_pct}%)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw CSV from run_old_driver.py")
    parser.add_argument("--output-clean", required=True, help="Cleaned CSV (no augmentation)")
    parser.add_argument("--output-augmented", required=True, help="Cleaned + augmented CSV")
    parser.add_argument("--accel-pct", type=float, default=7.0)
    parser.add_argument("--steer-pct", type=float, default=5.0)
    parser.add_argument("--brake-reduction-pct", type=float, default=10.0)
    args = parser.parse_args()

    print(f"[INFO] Loading {args.input}...")
    df_raw = pd.read_csv(args.input)
    print(f"[INFO] Raw shape: {df_raw.shape}")
    print(f"[INFO] Columns found: {list(df_raw.columns)}\n")

    # Passo 1: pulizia
    df_clean = clean(df_raw)
    df_clean.to_csv(args.output_clean, index=False)
    print(f"[OK] Clean CSV saved: {args.output_clean}  ({len(df_clean)} rows)")

    # Passo 2: aumenta solo gli output
    df_aug = augment(df_clean, args.accel_pct, args.steer_pct, args.brake_reduction_pct)
    df_aug.to_csv(args.output_augmented, index=False)
    print(f"[OK] Augmented CSV saved: {args.output_augmented}  ({len(df_aug)} rows)")

    print("\n[NEXT] Train BC with:")
    print("  conda run -n ai_env python scripts/train/train_bc_from_attempt1.py \\")
    print(f"    --original {args.output_clean} \\")
    print(f"    --augmented {args.output_augmented} \\")
    print("    --output-name bc_from_olddriver_v1")


if __name__ == "__main__":
    main()
