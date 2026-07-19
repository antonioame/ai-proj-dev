"""Driver principale: checkpoint CEM v5, promosso al posto di bc_tita_v20
perché più veloce (105.812s vs 114.02s/114.030s in due valutazioni dirette,
entrambi puliti: 0% fuori pista, 0 danni). Wrapper sottile su
drivers.cem.driver.CemDriver, per mantenere l'interfaccia BCDriver già usata
da run_agent.py/evaluate.py/record_agent.py.

Affidabilità: margine di stabilità più stretto del blend bc (3 giri puliti su
4 osservati alla promozione, vedi laptime_ledger.csv). bc_tita_v20 resta in
models/ per un rollback rapido in caso di problemi in uso reale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drivers.cem.driver import CemDriver

_CEM_V5_CHECKPOINT = Path(__file__).resolve().parents[1] / "drivers" / "rl" / "models" / "cem_v5.pth"


class BCDriver(CemDriver):
    """Il driver principale: alias di CemDriver(cem_v5), nome e interfaccia
    mantenuti per compatibilità con gli script esistenti (run_agent.py,
    evaluate.py, record_agent.py, ecc.), che istanziano BCDriver() senza
    argomenti."""

    def __init__(self):
        super().__init__(checkpoint_path=_CEM_V5_CHECKPOINT)
