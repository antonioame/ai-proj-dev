"""Driver — checkpoint CEM v5 (cross-entropy method, ottimizzazione black-box
a partire dai pesi del blend bc), promosso a driver di produzione il
2026-07-19 al posto del precedente modello singolo bc_tita_v20, verificato più
veloce (105.812s vs 114.02s/114.030s in due valutazioni comparative dirette,
stesse condizioni di test, entrambi puliti: 0% fuori pista, 0 danni).

Wrapper sottile attorno a drivers.cem.driver.CemDriver (stessa classe usata da
scripts/eval/evaluate_cem.py, già verificata) per mantenere invariata
l'interfaccia BCDriver già usata da run_agent.py/evaluate.py/record_agent.py.
Nota di affidabilità (vedi laptime_ledger.csv, tentativi cem_v3/v4/v5
rigettati): questa famiglia di checkpoint è nota per un margine di stabilità
più stretto del blend bc — su 4 valutazioni consecutive osservate durante la
promozione, 3 pulite e 1 con uscita di pista. bc_tita_v20 resta disponibile
in models/ (bc_tita_v20.*) per un rollback rapido se questa fragilità si
rivelasse un problema in uso reale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drivers.cem.driver import CemDriver

_CEM_V5_CHECKPOINT = Path(__file__).resolve().parents[1] / "drivers" / "rl" / "models" / "cem_v5.pth"


class BCDriver(CemDriver):
    """Alias di produzione per CemDriver(cem_v5) — nome/interfaccia mantenuti
    per compatibilità con gli script esistenti (run_agent.py, evaluate.py,
    record_agent.py, ecc.), che istanziano BCDriver() senza argomenti."""

    def __init__(self):
        super().__init__(checkpoint_path=_CEM_V5_CHECKPOINT)
