# torcs-ai — Agente AI per le corse in TORCS / Circuito Corkscrew

Un agente AI che guida il circuito Corkscrew in TORCS il più velocemente possibile,
costruito in tre fasi: baseline basato su regole → behavioral cloning → ottimizzazione RL.

## Architettura

```
Windows PC  ─── Server TORCS headless (UDP :3001)
                        │
                   Protocollo SCR (UDP)
                        │
MacBook Air M2 ─── Client Python + driver AI
```

---

## 1. Configurazione Windows (Server TORCS)

### 1a. Installa TORCS 1.3.x

Scarica l'installer di TORCS 1.3.7 da Windows dal sito ufficiale e installa in
`C:\torcs` (o in qualsiasi percorso senza spazi).

### 1b. Installa la patch SCR

La patch SCR (Simulated Car Racing) aggiunge una modalità server UDP a TORCS.

1. Scarica la patch SCR per TORCS 1.3.x.
2. Copia `scr_server.dll` (e i file correlati) in `C:\torcs\drivers\scr_server\`.
3. Verifica che TORCS trovi il driver: avvia TORCS normalmente e controlla l'elenco dei driver.

### 1c. Copia la configurazione di gara

Da questo repository, copia `torcs_env/race_config/corkscrew_solo.xml` in qualsiasi
posizione conveniente sulla macchina Windows (es. `C:\torcs\race_config\`).

### 1d. Avvia TORCS in modalità headless

```batch
cd C:\torcs
torcs.exe -r C:\torcs\race_config\corkscrew_solo.xml
```

TORCS si avvierà senza finestra, caricherà il circuito Corkscrew e attenderà un
client UDP sulla porta 3001.

> **Nota Firewall:** Consenti UDP in ingresso sulla porta 3001 nel Firewall Windows.
> Entrambe le macchine devono essere sulla stessa LAN (o instradare il traffico opportunamente).

---

## 2. Configurazione Mac M2 (Client Python)

### 2a. Ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2b. Installa PyTorch per Apple Silicon

```bash
# Build CPU (usa automaticamente MPS):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2c. Installa dipendenze del progetto

```bash
pip install -r requirements.txt
```

### 2d. Conferma che i test passano

```bash
pytest tests/ -v
# Atteso: 37 passed
```

---

## 3. Esecuzione dell'agente

### Avvio rapido

```bash
# Terminale 1 — Windows PC:
torcs.exe -r C:\torcs\race_config\corkscrew_solo.xml

# Terminale 2 — Mac (sostituisci con l'IP LAN del tuo Windows):
export TORCS_HOST=192.168.1.100
python scripts/run_agent.py --driver rule_based
```

L'agente si collegherà, completerà un giro, stamperà il tempo del giro ed uscirà.

### Opzioni

```
python scripts/run_agent.py --driver rule_based --laps 3 --host 192.168.1.100 --port 3001
```

---

## 4. Registrazione telemetria (Preparazione Fase 2)

```bash
export TORCS_HOST=192.168.1.100
python scripts/record_human.py --driver rule_based
# Salva: data/human_YYYYMMDD_HHMMSS.csv
```

Vedi `data/README.md` per lo schema CSV e la guida sulla qualità dei dati.

---

## 5. Valutazione

```bash
python scripts/evaluate.py --driver rule_based --laps 1
# Salva: results/eval_rule_based_YYYYMMDD_HHMMSS.json
```

Metriche riportate: tempo del giro, velocità massima, velocità media, % fuori pista, danno.

---

## 6. Fase 2 — Behavioral Cloning

Una volta che hai ≥5 giri puliti registrati:

```bash
python -m training.behavioral_cloning.train \
    --data data/human_*.csv \
    --output models/bc_v1.pth \
    --epochs 50
```

Poi implementa `drivers/bc/driver.py` che carica `models/bc_v1.pth` e passa
le osservazioni dei sensori attraverso la policy MLP.

---

## 7. Struttura del progetto

```
torcs_env/
  __init__.py
  client.py           # Client UDP (protocollo SCR)
  sensors.py          # Stringa sensori → dataclass SensorState
  actions.py          # Dataclass Action → stringa di controllo SCR
  race_config/
    corkscrew_solo.xml

drivers/
  base_driver.py      # BaseDriver astratto
  rule_based/
    driver.py         # Fase 1: controllo sterzo P + velocità PI

training/
  behavioral_cloning/
    dataset.py        # PyTorch Dataset da CSV
    model.py          # Rete policy MLP
    train.py          # Script di allenamento (consapevole di MPS)
  rl/
    README.md         # Piano Fase 3

scripts/
  run_agent.py        # Esegui un qualsiasi driver
  record_human.py     # Registra un giro su CSV
  evaluate.py         # Valutazione strutturata → output JSON

tests/                # 37 unit test (pytest)
data/                 # CSV telemetria (git-ignored)
results/              # File JSON valutazione (git-ignored)
```

---

## Risoluzione problemi

| Sintomo | Soluzione |
|---------|-----------|
| `ConnectionError: Could not connect to TORCS` | Controlla che TORCS sia in esecuzione e che `TORCS_HOST` / `TORCS_PORT` siano corretti; controlla Windows Firewall |
| TORCS esce immediatamente | Mismatch del nome modulo driver — modifica `corkscrew_solo.xml` → `<attstr name="module" val="scr_server"/>` |
| L'auto si schianta immediatamente | Sintonizza `STEER_ANGLE_GAIN` e `STEER_TRACK_GAIN` in `drivers/rule_based/driver.py` |
| `TimeoutError` dopo pochi secondi | TORCS ha perso la connessione; riavvia sia TORCS che l'agente |
| MPS non disponibile | Assicurati che PyTorch ≥ 2.1 e macOS ≥ 12.3; l'allenamento ricade automaticamente su CPU |
