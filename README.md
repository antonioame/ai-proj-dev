# torcs-ai — Agente AI per le corse in TORCS / Circuito Corkscrew

Agente AI che completa un giro del circuito Corkscrew in TORCS il più velocemente
possibile da una partenza da fermo, senza schiantarsi. Il driver principale
(candidato alla consegna finale) è un modello di **behavioral cloning**: **121.978 s**,
199.6 km/h di punta.

---

## Prerequisiti

- Windows con **TORCS 1.3.x** + patch **SCR** installate (il server di gara)
- Ambiente conda `ai_env` con le dipendenze installate:

```bash
conda create -n ai_env python=3.10
conda activate ai_env
pip install -r requirements.txt
```

`requirements.txt` non include PyTorch (va installato a parte, per piattaforma):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # con GPU NVIDIA
# oppure, solo CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Alcuni script accessori richiedono anche `Pillow` (livrea) e `joblib` (salvataggio
scaler/modelli in `_DRIVER/bc_source_driver/`) — installare se un import fallisce
(`pip install pillow joblib`).

---

## Eseguire il driver principale

```bash
# 1. Avvia il server TORCS in modalità headless
wtorcs.exe -r torcs_env\race_config\corkscrew_solo.xml

# 2. In un altro terminale, esegui il driver bc
conda run -n ai_env python scripts/run_agent.py --laps 1
```

L'agente si connette, guida il giro richiesto, stampa il tempo e salva un JSON
strutturato in `results/`.

**Opzioni utili:**

```bash
python scripts/run_agent.py --laps 3 --host localhost --port 3001 --telemetry
```

`--telemetry` salva anche il CSV completo dei sensori in `data/`.

**Solo valutazione** (stesse metriche, output più compatto in `results/eval_*.json`):

```bash
conda run -n ai_env python scripts/evaluate.py --laps 1
```

**Driver di riferimento isolato** (baseline basata su regole, ~148 s, non collegata
agli script principali):

```bash
conda run -n ai_env python old_versions_drivers/project_V2/run_rule_based.py --laps 1
```

---

## Livrea auto

Il progetto include una livrea personalizzata per `car1-ow1`. Tutti i comandi e i
dettagli sono in `CLAUDE.md` → "Configurazione livrea auto"; in breve:

```bash
conda run -n ai_env python livery/setup_livery.py                  # installa livery/car1-ow1.rgb
conda run -n ai_env python livery/setup_livery.py mia_livrea.png   # installa da PNG
conda run -n ai_env python livery/setup_livery.py --reset          # ripristina la livrea IBM originale
```

---

## Struttura del progetto

```
_DRIVER/            Driver in primo piano — candidato alla consegna
  driver.py             BCDriver, blend di due modelli
  models/               Modelli allenati (.pth/.npz)
  bc_source_driver/      Driver sorgente per rigenerare i dati di training
old_versions_drivers/project_V2/  Driver isolato, di solo riferimento (~148 s)
livery/               Risorse della livrea auto (car1-ow1)
torcs_env/            Protocollo SCR (client UDP, sensori, azioni, config gara)
scripts/              Entry point CLI (run_agent, evaluate, record_agent, ...)
tests/                Unit test (pytest, nessun server TORCS richiesto)
data/, results/       CSV telemetria e JSON di valutazione (git-ignored)
```

Per lo stato dettagliato di tutti i driver (attivi, isolati, rimossi) e le
decisioni progettuali, vedi `CLAUDE.md`.

---

## Risoluzione problemi

| Sintomo | Soluzione |
|---------|-----------|
| `ConnectionError: Could not connect to TORCS` | Verifica che TORCS sia in esecuzione e che `TORCS_HOST`/`TORCS_PORT` siano corretti (default `localhost:3001`) |
| TORCS esce subito dopo l'avvio | Mismatch nome modulo driver — controlla `corkscrew_solo.xml` → `<attstr name="module" val="scr_server"/>` |
| `TimeoutError` a metà corsa | TORCS ha perso la connessione; riavvia sia TORCS che lo script |
| PyTorch non trova la GPU | Verifica la build installata (CUDA vs CPU) per la tua macchina |

Esegui i test (nessun server TORCS richiesto):

```bash
conda run -n ai_env pytest tests/ -v
```
