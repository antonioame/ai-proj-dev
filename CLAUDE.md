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

### Fase C: Driver linea ottimale — IN PROGRESS (si schianta, necessita test)
- **Target: < 140 s** — follower della traiettoria con frenata tardiva
- Punto di ingresso: `python scripts/run_agent.py --driver optimal`
- Richiede `torcs_env/track_data/track_map.json` (già costruito dalla telemetria rule_based)
- **Sintonia conosciuta a partire da questa ristrutturazione:**
  - STARTUP_STEPS = 200 (fase di partenza da fermo conservativa, 4 secondi)
  - STEER_ANGLE_GAIN = 1.2 (era 1.6 — ridotto per prevenire tremolii)
  - STEER_LINE_GAIN = 0.25 (era 0.40 — tracciamento della linea meno aggressivo)
  - STEER_SMOOTH_SPEED = 75 (applica livellamento EMA fino a 75 km/h)
  - SCAN_AHEAD_M = 200 (era 300 — sguardo in avanti più focalizzato)
  - BRAKE_MARGIN_M = 40 (buffer di sicurezza extra sulla distanza di frenata)
  - TARGET_LINE_SCALE = 0.50 (miscela 50% verso la linea di gara, 50% centro)
  - TCS aggiunto (previene il pattinamento all'accelerazione)
- **Ricostruisci mappa** se registri nuova telemetria:
  ```bash
  python scripts/build_track_map.py --telemetry data/<file>.csv
  ```

### Rimossi (rotti, non ricreare senza un piano)
- **Fase 2 Behavioral Cloning** — si schiantato immediatamente; sterzo continuo, nessuna normalizzazione
- **Fase 3 Reinforcement Learning** — mismatch dello spazio di osservazione; eliminato

---

## Come eseguire

```bash
# 1. Avvia server TORCS (Windows)
torcs -r torcs_env/race_config/corkscrew_solo.xml

# 2. Esegui un driver (Mac o stessa macchina)
conda run -n ai_env python scripts/run_agent.py --driver rule_based
conda run -n ai_env python scripts/run_agent.py --driver optimal

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
  track_data/       track_map.json — precostruito dalla telemetria rule_based
drivers/
  base_driver.py    Interfaccia astratta
  registry.py       load_driver(name) — caricatore unico usato da tutti gli script
  rule_based/       Baseline Fase 1 (~148 s, stabile)
  optimal/          Follower traiettoria Fase C (in progress)
scripts/
  run_agent.py      Esegui un qualsiasi driver, opzionalmente salva telemetria + JSON risultati
  record_agent.py   Registra un giro su data/recorded_<driver>_<ts>.csv
  evaluate.py       Valuta e salva risultati strutturati JSON
  build_track_map.py  Costruisci track_map.json da CSV telemetria
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

Migliore attuale: **148.4 s** (rule_based, ABS + pressione freno più alta, commit ca54fea)

---

## Prossimi passi (ordinati per priorità)

1. **Test driver ottimale** — completa un giro senza schiantarsi?
   ```bash
   conda run -n ai_env python scripts/run_agent.py --driver optimal --laps 1
   ```
2. **Se ancora si schianta** — riduci ulteriormente `STEER_ANGLE_GAIN` (prova 1.0) o aumenta `BRAKE_MARGIN_M` (prova 60)
3. **Se stabile ma lento** — aumenta `CORNER_SPEED_SCALE` in `drivers/optimal/trajectory.py` (prova 1.1)
4. **Ricostruisci mappa tracciato** con più giri di telemetria per migliori stime della velocità in curva
