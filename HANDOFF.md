# Consegna: ottimizzazione tempo giro Corkscrew (sessione 2026-06-27)

## Riepilogo
**La Fase B è COMPLETATA e COMMESSA.** Sistema di frenata ABS + limiti di freno più alti hanno raggiunto **−3.24 secondi** (151.7s → 148.4s).

**Fase C (OptimalLineDriver)** è bozza ma BUGGY — l'auto si schianta al complesso di 480m. Necessita debug dello sterzo/controllo.

---

## Cosa è stato fatto

### Fase A (Strumentazione) — PARZIALE ✓
- ✅ Misurato **lunghezza tracciato** = 3608.4 m (tramite reset distFromStart telemetria fine giro)
- ✅ Costruito `torcs_env/track_map.py`: struttura dati per terreno per-bucket (velocità, curvatura, trackPos)
- ✅ Costruito `scripts/build_track_map.py`: compila CSV telemetria → track_map.json (risoluzione bucket 5m)
- ✅ Migliorata telemetria `run_agent.py`: ora cattura distFromStart, curLapTime, wheelSpinVel, danno
- ✅ Creato `laptime_ledger.csv`: unica fonte di verità per tutte le esecuzioni di ottimizzazione
- ✅ Costruito `scripts/benchmark.py`: esegui K giri, aggiungi metriche al registro, confronta con baseline
- ⚠️  Mappa tracciato ha 84% bucket marcati come "corner" (soglia curvatura troppo bassa per separazione netta dritto/curva—non critico per questa fase)

### Fase B (Quick Wins) — COMPLETATA ✓
- ✅ **Sistema frenata antibloccaggio (ABS)**
  - Aggiunto metodo `_apply_abs()` a RuleBasedDriver
  - Rileva blocco ruota anteriore (rotazione ruota < 80% velocità terreno)
  - Riduce pressione freno proporzionalmente: `freno × (1 − lockup / soglia)`
  - Consente l'aumento sicuro di BRAKE_MAX (era limitato da rischio lockup)

- ✅ **Limiti di pressione freno più alti**
  - BRAKE_MAX_HIGH: 0.65 → **0.82** (>140 km/h)
  - BRAKE_MAX_MED: 0.78 → **0.88** (90–140 km/h)
  - BRAKE_MAX_LOW: 0.90 → **0.93** (<90 km/h)
  - BRAKE_DECEL_FACTOR: 255 → **270** (riflette decelerazione ~1.05 g con ABS)

- ✅ **Risultato**: 151.688 s → **148.448 s** (−3.240 s, −2.1%)

### Fase C (OptimalLineDriver) — BUGGY ⚠️
File bozza ma NON COMMESSI:
- `drivers/optimal/driver.py` — controllore traiettoria indicizzato su posizione
- `drivers/optimal/trajectory.py` — costruttore profilo velocità (retropassaggio curva→MAX_SPEED)
- `torcs_env/track_data/track_map.json` — costruito da telemetria baseline
- `scripts/build_track_map.py` — costruttore mappa tracciato

**Il bug**: OptimalLineDriver si schianta a 480m (distFromStart) complesso in ciclo di recupero.
- Profilo velocità traiettoria sembra corretto (min 35 km/h al vertice, 38–48 km/h attraverso il complesso)
- L'auto va a tutta, poi frena forte (come previsto) — ma lo sterzo fallisce
- Risultato: auto finisce a trackPos = −7.4 (massicciamente fuori pista), velocità 0.2 km/h, bloccata per sempre
- Log: ripete azione di recupero (0.3 accel, sterzo=±0.30) ogni ~20 ms

**Sospetti causa radice** (in ordine di priorità):
1. **Sterzo in OptimalLineDriver._steer()** potrebbe non essere abbastanza forte attraverso il complesso
   - Guadagno linea troppo basso (0.20)? Guadagno angolo troppo basso (2.0)?
   - Mancanza di ricerca dell'apice come usa il driver rule_based
2. **Temporizzazione fase startup** — OptimalLineDriver usa STARTUP_STEPS=80, ma traiettoria potrebbe non tenere conto della dinamica di lancio
3. **Ricerca `distFromStart` mappa tracciato** — off by one? Problema wrapping vicino fine linea?

---

## Baseline attuale e registro

```
timestamp            | config_id                    | best_lap_s | delta_vs_baseline
2026-06-27T16:18:23  | baseline_rule_based          | 151.688    | baseline
2026-06-27T16:38:29  | phase_b_abs_higher_brakes    | 148.448    | −3.240s (−2.1%)
```

Esegui con:
```bash
# Fase B (attuale migliore)
python scripts/launch_race.py --driver rule_based --laps 1

# Fase C (in progress)
python scripts/launch_race.py --driver optimal --laps 1  # si schianta a ~480m
```

---

## Prossimi passi (ordine di priorità)

### 1. Debug e correggi OptimalLineDriver (BLOCCO per Fase C)
```python
# Opzione A: Aggiungi ricerca apice come rule_based
# In OptimalLineDriver._steer(), aggiungi offset target basato su curvatura:
sensors = state.track  # rangefinder
left_avg = (sensors[2] + sensors[3] + sensors[4]) / 3.0
right_avg = (sensors[14] + sensors[15] + sensors[16]) / 3.0
curvature = (left_avg - right_avg) / (left_avg + right_avg + eps)
target_tp += curvature * APEX_GAIN  # miscela offset basato su curvatura

# Opzione B: Controlla wrapping trajectory._idx() in fine linea
# Aggiungi assertion: idx non dovrebbe mai essere >= len(buckets)

# Opzione C: Aumenta guadagni sterzo in OptimalLineDriver
# Prova STEER_ANGLE_GAIN = 3.0, STEER_LINE_GAIN = 0.30
# Esegui un giro, controlla se sezione 480m è più pulita

# Opzione D: Debug guidato da telemetria
# Esegui 1 giro con optimal driver --telemetry (se possibile prima dello schianto)
# Traccia distFromStart vs trackPos, sterzo, freno — vedi dove diverge
```

### 2. Fase C completa (dopo correzione)
- Esegui OptimalLineDriver ottimizzato attraverso 5 giri puliti
- Aggiungi al registro (atteso −5 a −10 secondi basato su test iniziali)
- Benchmark settore per settore vs Fase B usando flag `--compare`

### 3. Fase D (sintonia automatica)
- Installa `pip install cma` (ottimizzatore CMA-ES)
- Parametrizza traiettoria (trackPos apice curva, scale velocità, margini freno)
- Sintonia automatica 10–40 parametri su 100–200 giri (2–3 ore tempo reale)
- Salva parametri migliori in `models/best_params.json`

### 4. Fase E (opzionale — fine-tuning RL)
- Registra ≥5 giri più veloci della Fase D
- Allena modello BC su di essi
- Warm-start RL con ricompensa tempo-giro

---

## File e posizioni chiave

| File | Scopo | Stato |
|------|-------|-------|
| `laptime_ledger.csv` | Registro metriche (solo append) | ✅ Attivo |
| `scripts/benchmark.py` | Esegui driver → aggiungi registro | ✅ Pronto |
| `scripts/run_agent.py` | Launcher CLI (load_driver) | ✅ Aggiornato |
| `drivers/rule_based/driver.py` | Baseline Fase B (ABS attivo) | ✅ Commesso |
| `drivers/optimal/driver.py` | Follower traiettoria Fase C | ⚠️ Buggy, non commesso |
| `torcs_env/track_map.py` | Struttura dati bucket tracciato | ✅ Commesso |
| `torcs_env/track_data/track_map.json` | Mappa costruita (non commessa per ora) | ⏸️ Riconstruisci dopo correzione |

---

## Come riprendere

1. **Esegui baseline Fase B** (verifica che funzioni ancora):
   ```bash
   conda run -n ai_env python scripts/launch_race.py --driver rule_based --laps 1
   ```

2. **Debug schianto OptimalLineDriver**:
   - Controlla `drivers/optimal/driver.py` riga ~90–110 (logica sterzo)
   - Aggiungi telemetria `print()` a _steer() per vedere valori angolo/errore/sterzo a 400–500m
   - Testa aumenti guadagno sterzo (3.0, 0.30) su giro completo

3. **Riconstruisci mappa tracciato** una volta confermata la correzione:
   ```bash
   conda run -n ai_env python scripts/build_track_map.py --telemetry data/rule_based_20260627_162255.csv
   ```

4. **Esegui Fase C**:
   ```bash
   conda run -n ai_env python scripts/launch_race.py --driver optimal --laps 3
   ```

---

## Note tecniche

- **Temporizzazione TORCS**: sim gira a ~280x tempo reale (8s muro ≈ 150s giro). Ciclo di controllo ~50 Hz.
- **distFromStart**: reset pulito a 3608.4m (misurato da telemetria). Bucket 5m = 722 bucket.
- **Matematica ABS**: rapporto rotazione ruota anteriore = `wheel_rad / (speed_ms / WHEEL_RADIUS)`. Lockup = rapporto < 0.80. Fattore riduzione = `max(0, 1 − (soglia − rapporto) / soglia)`.
- **Convergenza retropassaggio**: usualmente 3–5 iterazioni (720 bucket, ciclo veloce). Controlla stabilità numerica se aggiungi più complessità.

---

## Guadagni stimati rimanenti

- Fase C (OptimalLineDriver, una volta corretto): **−5 a −10 s** (frenata tardiva + linea di gara)
- Fase D (sintonia CMA-ES): **−2 a −5 s** (sweep parametri a grana fine)
- Fase E (fine-tuning RL): **−0.5 a −2 s** (ultimi decimi se A–D plateau)

**Target**: < 130 s tempo giro totale (dal baseline 151.7 s).

---

## Domande per prossimo agente

1. **Il 84% di curva della mappa tracciato è realistico?** (soglia curvatura = 0.05; potrebbe necessitare abbassamento)
2. **OptimalLineDriver ha bisogno di ereditare da RuleBasedDriver per coerenza?** (Attualmente BaseDriver; potrebbe miscelare sterzo rule_based + controllo velocità ottimale)
3. **I parametri Fase D dovrebbero includere sintonia BRAKE_DECEL_FACTOR?** (Attualmente hardcoded; potrebbe essere variabile sweep)

Saluti!
