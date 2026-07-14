"""Filtra e deduplica il dataset DAgger grezzo (scripts/record_dagger.py) prima
del training BC.

Due passi:
1. Rimuove le righe "degenerate": fuori pista (|trackPos| > 1.0) OPPURE quasi
   ferme (speed < 5 km/h) — unione dei due criteri, non solo trackPos, perché
   la coda "fermo al muro" satura anche righe ancora on-track a v~0 prima di
   uscire fisicamente dai bordi.
2. Su quanto resta, comprime le sequenze quasi-identiche consecutive (stesso
   stato/azione ripetuto per molti tick, tipico di micro-recovery in curva)
   a un tetto di MAX_PER_SEGMENT campioni per segmento, per non far dominare
   il gradiente di training con lo stesso istante ripetuto migliaia di volte
   — senza però buttare via il segnale di recovery del tutto.

Input: uno o più CSV prodotti da record_dagger.py (colonne feat_0..feat_25 +
oracle_steer/accel/brake + rollout_* + lap/distFromStart).
Output: un unico CSV filtrato, più un riepilogo stampato a schermo.

Usage:
    python scripts/filter_dagger_dataset.py data/dagger_bc_20260714_152520.csv data/dagger_bc_run2.csv --out data/dagger_bc_filtered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPEED_MIN_KMH = 5.0
TRACKPOS_MAX = 1.0
MAX_PER_SEGMENT = 30  # tetto di campioni per segmento quasi-identico consecutivo

# Soglie di "quasi-identico" tra righe consecutive (stesse unità delle colonne)
DUP_THRESH = {
    "feat_1": 1.0,    # speed (km/h)
    "feat_4": 0.01,   # trackPos
    "feat_0": 0.01,   # angle (rad)
    "oracle_steer": 0.01,
    "oracle_accel": 0.01,
    "oracle_brake": 0.01,
}


def _is_near_duplicate(prev: pd.Series, cur: pd.Series) -> bool:
    return all(abs(cur[col] - prev[col]) < thresh for col, thresh in DUP_THRESH.items())


def _collapse_near_duplicate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Comprime le run consecutive quasi-identiche a MAX_PER_SEGMENT campioni,
    sottocampionati in modo uniforme lungo la run.
    """
    keep_mask = np.zeros(len(df), dtype=bool)
    run_start = 0
    rows = df.reset_index(drop=True)

    def _flush(start: int, end: int) -> None:
        # [start, end) run di righe quasi-identiche consecutive
        run_len = end - start
        if run_len <= MAX_PER_SEGMENT:
            keep_mask[start:end] = True
        else:
            idx = np.linspace(start, end - 1, MAX_PER_SEGMENT).round().astype(int)
            keep_mask[np.unique(idx)] = True

    for i in range(1, len(rows)):
        if not _is_near_duplicate(rows.iloc[i - 1], rows.iloc[i]):
            _flush(run_start, i)
            run_start = i
    _flush(run_start, len(rows))

    return rows[keep_mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and dedupe raw DAgger CSVs before BC training")
    parser.add_argument("inputs", nargs="+", help="Raw DAgger CSV file(s)")
    parser.add_argument("--out", required=True, help="Output filtered CSV path")
    args = parser.parse_args()

    dfs = []
    for f in args.inputs:
        df = pd.read_csv(f)
        df["source"] = Path(f).name
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    n_raw = len(raw)

    degenerate = (raw["feat_4"].abs() > TRACKPOS_MAX) | (raw["feat_1"].abs() < SPEED_MIN_KMH)
    clean = raw[~degenerate].reset_index(drop=True)
    n_after_degenerate_filter = len(clean)

    # Comprimi le run quasi-identiche separatamente per ogni file/giro sorgente,
    # così non si fondono run di sessioni diverse.
    collapsed_parts = []
    for (source, lap), group in clean.groupby(["source", "lap"], sort=False):
        collapsed_parts.append(_collapse_near_duplicate_runs(group))
    final = pd.concat(collapsed_parts, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)

    print(f"Righe grezze totali:                    {n_raw}")
    print(f"Rimosse (|trackPos|>1.0 OR speed<5km/h): {n_raw - n_after_degenerate_filter}")
    print(f"Dopo filtro degenerate:                  {n_after_degenerate_filter}")
    print(f"Rimosse da compressione run duplicate:   {n_after_degenerate_filter - len(final)}")
    print(f"RIGHE FINALI:                            {len(final)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
