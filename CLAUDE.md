# TORCS AI — Contesto del progetto per sessioni Claude Code

## Obiettivo

Allenare un agente AI per completare un singolo giro del circuito **Corkscrew** in TORCS
il più velocemente possibile da una partenza da fermo, senza schiantarsi.

**Metrica di successo:** Tempo del giro (più basso è meglio).  
**Vincoli:** Nessuno schianto, escursioni minime fuori pista.

---

## Configurazione hardware

| Macchina | Ruolo | Note |
|---------|-------|-------|
| Windows PC | Server TORCS headless | Esegue `torcs -r`, porta UDP 3001 |
| MacBook Air M2 | Client Python + allenamento | PyTorch con backend MPS |

Entrambe le macchine sono sulla stessa LAN. Il Mac si connette al server UDP TORCS
tramite la variabile d'ambiente `TORCS_HOST=<windows-LAN-IP>`.

Variabili d'ambiente principali:
```
TORCS_HOST   (default: localhost)
TORCS_PORT   (default: 3001)
```

---

## Configurazione livrea auto

Il progetto include una livrea auto personalizzata (`livrea.png`) applicata in modo sicuro e reversibile.

**Installa livrea:**
```bash
conda run -n ai_env python scripts/setup_livery.py --install
```

**Controlla stato:**
```bash
conda run -n ai_env python scripts/setup_livery.py --status
```

**Ripristina originale (completamente reversibile):**
```bash
conda run -n ai_env python scripts/setup_livery.py --rollback
```

**Come funziona:**
- Converte `livrea.png` (PNG) → formato RGB Radiance (nativo TORCS)
- Applica alla texture dell'auto `car1-stock1`
- Backup automatico dell'originale `car1-stock1.rgb` in `.rgb.backup`
- Può essere ripristinato all'originale senza alcuna perdita

---

## Stato dei driver

### Fase 1: Basato su regole — COMPLETATO ✓ (baseline stabile)
- **Tempo giro: ~148 s**, nessuno schianto
- Punto di ingresso: `python scripts/run_agent.py --driver rule_based`
- Sintonizzato con ABS, TCS, ricerca dell'apice, controllo PI della spinta
- Vedi `drivers/rule_based/driver.py` per tutte le costanti

### Driver BC ibrido — CANDIDATO ALLA CONSEGNA ✓
- **Tempo giro: 125.790 s**, top speed 199.0 km/h (test del 2026-07-01, commit bcfe1f9) — migliore di rule_based
- Punto di ingresso: `python scripts/run_agent.py --driver bc`
- Blend di due modelli: `models/bc_from_rulefriend_v1.*` (rettilineo) + `models/bc_from_olddriver_v1.*` (curva)
- `bc_from_rulefriend_v1` è stato allenato su telemetria generata da `bc_source_driver/run_friend_model.py`
  (driver sorgente conservato in una cartella separata in root — riesegui quello script per nuovi campioni)
- Vedi `drivers/bc/driver.py` per i dettagli del blend

### Rimossi (rotti, non ricreare senza un piano)
- **Fase 2 Behavioral Cloning (versione iniziale)** — si schiantato immediatamente; sterzo continuo, nessuna normalizzazione
- **Fase 3 Reinforcement Learning** — mismatch dello spazio di osservazione; eliminato
- **Fase C Driver linea ottimale** (`drivers/optimal/`) — non funzionante in pista, rimosso insieme a `scripts/build_track_map.py`, `torcs_env/track_map.py`, `torcs_env/track_data/`, e i relativi doc in `docs/`
- **Progetti degli amici** (`old_project_material/project_made_by_my_friend/`, `_V2/`, `old_project_material/Friends_Projects/`) — testati uno per uno, tutti non funzionanti (modelli mancanti, import rotti, dataset mancanti). Rimossi interamente. L'unica parte "amico-derivata" ancora in uso è `bc_source_driver/` (vedi sopra), conservata perché serve a rigenerare i dati di training del driver `bc`.

---

## Come eseguire

```bash
# 1. Avvia server TORCS (Windows)
torcs -r torcs_env/race_config/corkscrew_solo.xml

# 2. Esegui un driver (Mac o stessa macchina)
conda run -n ai_env python scripts/run_agent.py --driver rule_based

# 3. Registra telemetria
conda run -n ai_env python scripts/record_agent.py --driver rule_based

# 4. Valuta (salva JSON in results/)
conda run -n ai_env python scripts/evaluate.py --driver rule_based --laps 1
```

---

## Decisioni progettuali principali

| Decisione | Razionale |
|-----------|-----------|
| Solo client UDP (nessun plugin TORCS) | La patch SCR espone un'interfaccia UDP pulita; nessun C++ necessario |
| Rilevamento reset `distRaced` per conteggio giri | `lastLapTime` aggiorna solo una volta per giro; distRaced è continuo |
| `drivers/registry.py` per caricamento driver | Unica fonte di verità — run_agent, record_agent, evaluate la usano tutti |
| Target di velocità basato su fisica in rule_based | Formula di distanza di frenata, non tabella di ricerca — nessuna discontinuità |
| ABS su entrambi i driver | Previene il bloccaggio con valori alti di BRAKE_MAX |
| TCS su entrambi i driver | Previene il pattinamento all'accelerazione |
| Traiettoria con retropassaggio | Propaga i limiti di velocità delle curve all'indietro per impostare i punti di frenata |
| `TARGET_LINE_SCALE = 0.50` | Miscela la linea di gara con il centro per ridurre il rischio di uscite di pista |

---

## Layout del repository

```
torcs_env/          Protocollo SCR (sensori, azioni, client UDP, XML gara)
drivers/
  base_driver.py    Interfaccia astratta
  registry.py       load_driver(name) — caricatore unico usato da tutti gli script
  rule_based/       Baseline Fase 1 (~148 s, stabile)
  bc/                Behavioral cloning ibrido (125.8 s, candidato consegna)
bc_source_driver/    Driver sorgente usato per generare i dati di bc_from_rulefriend_v1
                      (rieseguire run_friend_model.py per nuovi campioni)
scripts/
  run_agent.py      Esegui un qualsiasi driver, opzionalmente salva telemetria + JSON risultati
  record_agent.py   Registra un giro su data/recorded_<driver>_<ts>.csv
  evaluate.py       Valuta e salva risultati strutturati JSON
tests/              Unit test
data/               CSV telemetria (git-ignored)
results/            File JSON valutazione (git-ignored)
laptime_ledger.csv  Log manuale di esperimenti di sintonia
```

---

## Registro tempo giro

Registra ogni esecuzione di benchmark in `laptime_ledger.csv`:
```
timestamp,config_id,git_sha,best_lap_s,median_lap_s,top_speed_kmh,off_track_pct,damage,valid,notes
```

Migliore attuale: **125.790 s** (bc, hybrid rulefriend/olddriver, commit bcfe1f9)

---

## Prossimi passi

Driver `optimal` e tutti i progetti degli amici non funzionanti sono stati rimossi. `bc` è il candidato alla consegna (125.8 s, batte rule_based). Prossimo passo: confermare la stabilità di `bc` su più giri prima della consegna finale.
