# Eagles Racing Team - Corkscrew AI Driver

Agente AI che completa un giro del circuito Corkscrew in TORCS il più velocemente
possibile da una partenza da fermo, senza schiantarsi. Il driver principale (candidato
alla consegna finale) è un checkpoint ottimizzato con **CEM** (cross-entropy method,
black-box a partire dai pesi BC), `cem_v5`: **105.812 s**, 187.1 km/h di punta, 0% fuori
pista, 0 danni: record assoluto del progetto, promosso al posto del precedente driver
principale, un modello di **behavioral cloning** a modello singolo clonato dallo stile
di guida del bot nativo `tita` (`bc_tita_v20`, 111.986 s, ancora disponibile in
`_DRIVER/models/` per rollback).

---

## Prerequisiti

- Windows con **TORCS 1.3.x** + patch **SCR** installate (il server di gara)
- Ambiente conda `ai_env` con le dipendenze installate:

```bash
conda create -n ai_env python=3.10
conda activate ai_env
pip install -r requirements.txt
```

---

## Eseguire il driver principale

```bash
# 1. Avvia il server TORCS in modalità headless [opzionale]
wtorcs.exe -r torcs_env\race_config\corkscrew_solo.xml

# 2. In un altro terminale, esegui il driver principale
conda run -n ai_env python scripts/run/run_agent.py --laps 1
```

L'agente si connette, guida il giro richiesto, stampa il tempo e salva un JSON
strutturato in `results/`.

**Opzioni utili:**

```bash
conda run -n ai_env python scripts/run/run_agent.py --laps 3 --host localhost --port 3001 --telemetry
```

`--telemetry` salva anche il CSV completo dei sensori in `data/`.

**Solo valutazione** (stesse metriche, output più compatto in `results/eval_*.json`):

```bash
conda run -n ai_env python scripts/eval/evaluate.py --laps 1
```

---

## Livrea auto

Il progetto include una livrea personalizzata per `car1-ow1`. Tutti i comandi e i
dettagli sono qui sotto; in breve:

```bash
conda run -n ai_env python livery/setup_livery.py                  # installa livery/car1-ow1.rgb
conda run -n ai_env python livery/setup_livery.py mia_livrea.png   # installa da PNG
conda run -n ai_env python livery/setup_livery.py --reset          # ripristina la livrea IBM originale
```

---

## Struttura del progetto

```
_DRIVER/                Driver principale
  driver.py             BCDriver, wrapper su drivers/cem/driver.py:CemDriver (cem_v5)
  models/               Modelli allenati (.pth/.npz; incl. bc_tita_v20 e il vecchio blend, per rollback)
  bc_source_driver/     Driver sorgente per rigenerare i dati di training
drivers/                Driver aggiuntivi (cem/ è la sorgente del driver principale; bc_dagger, rl residual non promossi) + bc_common.py
training/rl/            Fase 3: Infrastruttura RL/CEM (env Gymnasium, reward, training)
data_collection/tita/   Pipeline di clonazione del bot tita (conversione CSV, DAgger-style)
livery/                 Risorse della livrea auto (car1-ow1)
torcs_env/              Protocollo SCR (client UDP, sensori, azioni, config gara)
scripts/                Entry point CLI (run_agent, evaluate, record_agent, ...)
tests/                  Unit test (pytest, nessun server TORCS richiesto)
data/                   CSV telemetria (i dataset di training BC sono git-tracked)
results/                JSON di valutazione (git-ignored)
```
