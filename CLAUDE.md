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

## Configurazione livrea auto (solo car1-ow1)

Tutte le risorse della livrea vivono in `livery/` (livello radice): immagini, script di
installazione/reset, decoder di debug, file di stato. Lo script gestisce **solo car1-ow1**
(unica auto usata in gioco) nel vero formato SGI RGB non compresso 512×512 RGBA (verificato
byte-per-byte contro il file `car1-ow1.rgb` di gioco — non è il formato "Radiance" usato in
una versione precedente per car1-stock1, ormai rimossa).

**Installa una nuova livrea da PNG** (converte e salva come `livery/car1-ow1.rgb`, poi installa in TORCS):
```bash
conda run -n ai_env python livery/setup_livery.py livery/eagles_livery.png
```

**Reinstalla la livrea già pronta in `livery/car1-ow1.rgb`** (nessuna conversione, nessun argomento):
```bash
conda run -n ai_env python livery/setup_livery.py
```

**Ripristina la livrea originale IBM** (rigenera `livery/original_IBM_livery/car1-ow1.rgb`
da `livery/original_IBM_livery/original_IBM_livery.png` e la installa):
```bash
conda run -n ai_env python livery/setup_livery.py --reset
```

**Controlla stato:**
```bash
conda run -n ai_env python livery/setup_livery.py --status
```

**Ripristina l'ultimo backup lato TORCS** (qualunque cosa fosse installata prima dell'ultima `install`):
```bash
conda run -n ai_env python livery/setup_livery.py --rollback
```

**Debug/anteprima** (decodifica `.rgb` → PNG per ispezione visiva):
```bash
conda run -n ai_env python livery/decode_sgi.py
conda run -n ai_env python livery/decode_rgb.py
```

**Come funziona:**
- Da PNG: converte in formato SGI RGB 512×512 RGBA non compresso (4 piani R,G,B,A) e salva il risultato in `livery/car1-ow1.rgb`
- Applica alla texture dell'auto `car1-ow1`
- Backup automatico dell'originale `car1-ow1.rgb` (lato TORCS) in `.rgb.backup`
- Reset IBM e rollback da backup sono entrambi completamente reversibili e indipendenti

---

## Stato dei driver

### Driver BC ibrido — CANDIDATO ALLA CONSEGNA, IN PRIMO PIANO ✓
- **Tempo giro: 125.790 s**, top speed 199.0 km/h (test del 2026-07-01, commit bcfe1f9) — migliore di rule_based
- Punto di ingresso: `python scripts/run_agent.py --laps 1` (nessun `--driver`, è l'unico driver collegato agli script)
- Tutte le risorse vivono in `_DRIVER/` (livello radice): `driver.py`, `models/` (i due modelli del blend),
  `bc_source_driver/` (driver sorgente usato per generare i dati di training, riesegui per nuovi campioni)
- Blend di due modelli: `_DRIVER/models/bc_from_attempt1_v1.*` (rettilineo, da un tentativo precedente di
  driving-net) + `_DRIVER/models/bc_from_olddriver_v1.*` (curva)

### Fase 1: Basato su regole — ISOLATO (di riferimento, non più in primo piano)
- **Tempo giro: ~148 s**, nessuno schianto — ma più lento del driver bc
- Spostato interamente in `old_versions_drivers/project_V2/` (livello radice), **non più collegato** a
  `scripts/run_agent.py`/`registry.py` (rimosso)
- Punto di ingresso dedicato: `python old_versions_drivers/project_V2/run_rule_based.py --laps 1`

### Rimossi (rotti, non ricreare senza un piano)
- **Fase 2 Behavioral Cloning (versione iniziale)** — si schiantato immediatamente; sterzo continuo, nessuna normalizzazione
- **Fase 3 Reinforcement Learning** — mismatch dello spazio di osservazione; eliminato
- **Fase C Driver linea ottimale** (`drivers/optimal/`) — non funzionante in pista, rimosso insieme a `scripts/build_track_map.py`, `torcs_env/track_map.py`, `torcs_env/track_data/`, e i relativi doc in `docs/`
- **Tentativi di driving agent esterni** (`old_project_material/project_made_by_my_friend/`, `_V2/`, `old_project_material/Friends_Projects/`) — vecchie implementazioni di terzi testate una per una, tutte non funzionanti (modelli mancanti, import rotti, dataset mancanti). Rimosse interamente. L'unico tentativo precedente ancora in uso è il driving-net in `_DRIVER/bc_source_driver/attempt_model/` (vedi sopra), conservato perché serve a rigenerare i dati di training del driver `bc`.
- **Vecchio driver personale** (`old_project_material/torcs_jm_par.py`) — testato: tempo 123.0 s, ma marce che vanno a limitatore in 1ª/2ª; tenuto solo come riferimento, non integrato negli script.

---

## Come eseguire

```bash
# 1. Avvia server TORCS (Windows)
torcs -r torcs_env/race_config/corkscrew_solo.xml

# 2. Esegui il driver bc (Mac o stessa macchina)
conda run -n ai_env python scripts/run_agent.py --laps 1

# 3. Registra telemetria
conda run -n ai_env python scripts/record_agent.py --laps 1

# 4. Valuta (salva JSON in results/)
conda run -n ai_env python scripts/evaluate.py --laps 1

# (riferimento, isolato) rule_based archiviato
conda run -n ai_env python old_versions_drivers/project_V2/run_rule_based.py --laps 1
```

---

## Decisioni progettuali principali

| Decisione | Razionale |
|-----------|-----------|
| Solo client UDP (nessun plugin TORCS) | La patch SCR espone un'interfaccia UDP pulita; nessun C++ necessario |
| Rilevamento reset `distRaced` per conteggio giri | `lastLapTime` aggiorna solo una volta per giro; distRaced è continuo |
| `_DRIVER/driver.py` importato direttamente (no registry) | Un solo driver in uso — l'indirezione registry/`--driver` è stata rimossa quando rule_based è stato isolato |
| Target di velocità basato su fisica in rule_based | Formula di distanza di frenata, non tabella di ricerca — nessuna discontinuità |
| ABS su entrambi i driver | Previene il bloccaggio con valori alti di BRAKE_MAX |
| TCS su entrambi i driver | Previene il pattinamento all'accelerazione |
| Traiettoria con retropassaggio | Propaga i limiti di velocità delle curve all'indietro per impostare i punti di frenata |
| `TARGET_LINE_SCALE = 0.50` | Miscela la linea di gara con il centro per ridurre il rischio di uscite di pista |

---

## Layout del repository

```
torcs_env/          Protocollo SCR (sensori, azioni, client UDP, XML gara)
_DRIVER/           Driver IN PRIMO PIANO — candidato alla consegna (125.8 s)
  driver.py          BCDriver, blend di due modelli
  models/             bc_from_attempt1_v1.*, bc_from_olddriver_v1.*
  bc_source_driver/   Driver sorgente (tentativo precedente) per rigenerare i dati di bc_from_attempt1_v1
old_versions_drivers/project_V2/  Driver ISOLATO, di solo riferimento (~148 s, non in registry/run_agent)
  driver.py
  run_rule_based.py   Script minimale per eseguirlo standalone
livery/               Tutte le risorse della livrea auto (immagini, setup/rollback, decoder debug)
scripts/
  run_agent.py      Esegue il driver bc, opzionalmente salva telemetria + JSON risultati
  record_agent.py   Registra un giro su data/recorded_bc_<ts>.csv
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
config_id,git_sha,best_lap_s,median_lap_s,top_speed_kmh,off_track_pct,damage,valid,notes
```

Migliore attuale: **125.790 s** (bc, hybrid attempt1/olddriver, commit bcfe1f9)

---

## Prossimi passi

Repository riorganizzato attorno al driver `bc` (in `_DRIVER/`, in primo piano, candidato alla consegna).
`rule_based` è isolato in `old_versions_drivers/project_V2/`, scollegato dagli script principali. Prossimo passo:
confermare la stabilità di `bc` su più giri prima della consegna finale.
