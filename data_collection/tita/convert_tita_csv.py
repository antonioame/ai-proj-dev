"""Converte i CSV grezzi prodotti dal logger C++ di tita (bc_logs/tita_bc_*.csv)
nel formato del dataset BC esistente (stesse colonne di data/driver_*.csv).

Il logger C++ scrive gia' le colonne nell'ordine e nomenclatura corretti
(vedi bc_logger.cpp), quindi qui si valida lo schema, si scartano le righe
non utilizzabili (stessa pulizia applicata da scripts/train/train_bc_dagger.py:
|trackPos| >= 0.95, |speed| <= 1.0) e si scrive un nuovo file per sessione
in data_collection/tita/converted/, senza toccare data/ ne' i CSV originali.

Usage:
    python data_collection/tita/convert_tita_csv.py --input "bc_logs/tita_bc_*.csv"
    python data_collection/tita/convert_tita_csv.py --input "bc_logs/tita_bc_1_20260715_120000.csv" --no-clean
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

EXPECTED_COLS = (
    ["angle", "speed", "speedY", "speedZ", "trackPos"]
    + [f"track_{i}" for i in range(19)]
    + ["rpm", "gear", "steer", "accel", "brake", "gear_cmd"]
)

OUTPUT_DIR = Path(__file__).parent / "converted"


def convert_one(path: str, clean: bool) -> None:
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: colonne mancanti rispetto allo schema BC atteso: {missing}")

    df = df[EXPECTED_COLS]
    n_raw = len(df)

    if clean:
        df = df[df["trackPos"].abs() < 0.95]
        df = df[df["speed"].abs() > 1.0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"tita_bc_{Path(path).stem}.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] {path}: {n_raw} righe grezze -> {len(df)} righe scritte in {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path o glob dei CSV grezzi prodotti da tita (es. bc_logs/tita_bc_*.csv)")
    parser.add_argument("--no-clean", action="store_true", help="Non filtrare le righe fuori pista / ferme")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise FileNotFoundError(f"Nessun file corrisponde a: {args.input}")

    for f in files:
        convert_one(f, clean=not args.no_clean)

    print(
        f"\n[INFO] Conversione completata. Per usare questi dati in train_bc_dagger.py:\n"
        f'  python scripts/train/train_bc_dagger.py --original "{OUTPUT_DIR}/*.csv" --dagger data/dagger_bc_filtered.csv\n'
        f"oppure copia manualmente i file in data/ per unirli al dataset originale."
    )


if __name__ == "__main__":
    main()
