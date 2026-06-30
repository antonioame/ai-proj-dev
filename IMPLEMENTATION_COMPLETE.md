# Correzione sterzo zero: implementazione completa ✓

**Data:** 2026-06-29  
**Stato:** Tutte le correzioni applicate, modello allenato, pronto per test

---

## Cosa è stato fatto

### 1. Indagine causa radice
Identificato che il problema dello sterzo zero era causato da **mismatch di normalizzazione input** tra allenamento RL e BC:
- RL allenato su: divisori hardcoded `[speed/300, rpm/10k, ...]`
- BC allenato su: z-score normalizzato `[(speed - media) / std, ...]`
- Risultato: il modello non poteva imparare lo sterzo

### 2. Correzioni codice applicate (commit 727593b)

#### gym_env.py
- Cambio spazio di osservazione da 9 → 8 feature (rimosso danno)
- Cambio indici track da [7, 9, 11] → [6, 12, 18] (corrisponde BC)
- Aggiunta z-score normalizzazione usando statistiche BC v2.pth
- Nuove costanti normalizzazione hardcoded per coerenza

#### drivers/rl/driver.py
- Importa costanti normalizzazione condivise da gym_env
- Usa implementazione `_make_obs()` identica
- Garantisce simmetria allenamento/inferenza

#### training/rl/train_rl_bc_warmstart.py
- **Riscrittura completa** con 3 miglioramenti maggiori:
  1. **Costruisci modello PRIMA di avviare TORCS** (corregge timeout pre-connessione)
  2. **Corretta inizializzazione peso BC** (mappa backbone BC → policy_net PPO)
  3. **Conteggio step reale** (via callback SB3, non stime tempo trascorso)
- Step per sessione: 1000 (abbastanza basso per evitare timeout TORCS)

### 3. Test
- ✅ Tutti 37 unit test passano
- ✅ Allineamento osservazione verificato (gym_env = driver RL)
- ✅ Validazione checkpoint BC (8-dim, statistiche corrispondono)
- ✅ Modello PPO costruisce senza TORCS

### 4. Allenamento eseguito
```
Allenamento: RL fine-tuning con BC Warm-start (v3_fixed)
  Ora inizio: 2026-06-29 00:28:28
  Ora fine:   2026-06-29 00:35:46
  Durata:     ~7.3 minuti
  Sessioni:   37
  Step totali: 100,488 (target: 100,000)
  Output:     models/rl_bc_warmstart_v3_fixed/final.zip (1.6 MB)
```

---

## Cosa testare dopo

### Opzione A: singolo giro (test rapido)

```bash
# Avvia TORCS
wtorcs.exe -r torcs_env\race_config\corkscrew_solo.xml

# Esegui modello su 1 giro con telemetria
conda run -n ai_env python scripts/run_agent.py ^
    --driver rl_rl_bc_warmstart_v3_fixed ^
    --laps 1 ^
    --telemetry
```

**Atteso:**
- Valori sterzo **non-zero** nelle curve (non tutti zeri)
- Giro **completato** (nessuno schianto a ~3.2 km)
- Tempo giro registrato in `results/`

**Controlla CSV telemetria per:**
```
Colonna "steer": dovrebbe variare durante curve (non costante 0)
Colonna "speed": dovrebbe mostrare velocità realistiche
Conteggio righe: dovrebbe essere ~5000-10000 (giro completo)
```

### Opzione B: confronto prestazioni (5 giri)

```bash
# Baseline (rule-based): ~148 s/giro
conda run -n ai_env python scripts/run_agent.py --driver rule_based --laps 5

# RL migliorato (target: < 150 s/giro)
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v3_fixed --laps 5
```

---

## Metriche chiave

| Metrica | Prima | Dopo (atteso) |
|---------|-------|---------------|
| Output sterzo | Zero nelle curve | Non-zero, progressivo |
| Completamento giro | Schianto a 3.2 km | Giro completo |
| Dim osservazione | 9 (raw scaled) | 8 (z-scored) |
| Lookahead track | ±6° stretto | ±9° più ampio |
| Dimensione modello | — | 1.6 MB |
| Tempo allenamento | — | ~7 minuti (100k step) |

---

## File modificati

```
training/rl/gym_env.py              (correzione spazio osservazione)
drivers/rl/driver.py                (allineamento inferenza)
training/rl/train_rl_bc_warmstart.py (riscrittura pipeline allenamento)
INVESTIGATION_REPORT.md             (documento causa radice)
docs/ZERO_STEERING_SUMMARY.md       (sommario esecutivo)
docs/INVESTIGATION_ZERO_STEERING.md (approfondimento tecnico)
docs/FIX_ZERO_STEERING.md          (guida implementazione)
docs/RUNNING_DRIVERS.md            (how-to per tutti i modelli)
SOLVING_ZERO_STEERING.md           (documento approccio)
```

---

## Posizione modello allenato

```
models/rl_bc_warmstart_v3_fixed/
├── model.zip     [1.6 MB] - Ultimo checkpoint (sessione 37)
└── final.zip     [1.6 MB] - Checkpoint finale esportato
```

**Esegui con:**
```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v3_fixed --laps 1
```

---

## Checklist verificazione

- [x] Causa radice identificata (mismatch normalizzazione input)
- [x] Correzioni codice applicate (gym_env, driver, script allenamento)
- [x] Tutti unit test passano (37/37)
- [x] Allineamento osservazione verificato
- [x] Modello allenato (100k+ step)
- [x] Checkpoint salvato
- [ ] Modello testato su 1 giro (TU DOPO)
- [ ] Sterzo confermato non-zero (TU DOPO)
- [ ] Tempo giro misurato (TU DOPO)

---

## Se ancora non funziona

### Sintomo: sterzo ancora zero

1. **Controlla normalizzazione osservazione:**
   ```bash
   python -c "from training.rl.gym_env import _OBS_MEAN; print(_OBS_MEAN)"
   ```
   Dovrebbe stampare statistiche BC (88.3, -0.07, 0.008, ...), non zeri.

2. **Controlla indici track:**
   ```bash
   python -c "from training.rl.gym_env import _TRACK_IDX; print(_TRACK_IDX)"
   ```
   Dovrebbe stampare (6, 12, 18), non (7, 9, 11).

3. **Cerca log:**
   - Controlla `results/rl_rl_bc_warmstart_v3_fixed_*.json` per tempo giro
   - Controlla qualsiasi telemetria `.csv` per valori colonna sterzo

4. **Prova modello alternativo:**
   ```bash
   # Fallback al precedente migliore (50k step)
   conda run -n ai_env python scripts/run_agent.py --driver rl_bc_warmstart --laps 1
   ```

---

## Criteri di successo

Il modello funziona se:

1. ✅ **Sterzo non-zero:** colonna sterzo in CSV telemetria ha valori come 0.1, -0.2, 0.05 (non tutti 0)
2. ✅ **Giro completato:** modello gira ~5000+ step senza schianto
3. ✅ **Tempo giro < 150 s:** prestazioni a tiro di schioppo da rule-based (148.4 s)

---

## Fase successiva (dopo test)

Una volta che sterzo funziona:

1. **Misura guadagno tempo giro:**
   - Confronta v3_fixed vs rule_based su 5 giri
   - Target: entro 2-3% di rule-based (< 153 s)

2. **Considera raffinamenti:**
   - Lookahead track più ampio: cambia `_TRACK_IDX = (4, 9, 14)` per ±15°
   - Rete più grande: prova `[512, 512, 256]` se sterzo è debole
   - Più allenamento: estendi a 200k step se ancora migliorando

3. **Documenta risultati:**
   - Salva tempo giro migliore in `PERFORMANCE_LOG.md`
   - Registra quale variante driver ha performato meglio

---

## Riferimento commit

- **Commit:** 727593b
- **Branch:** main2
- **Data:** 2026-06-29
- **Modifiche:** 11 file modificati, 1342 inserimenti

## Riferimento allenamento

- **Script:** train_rl_bc_warmstart.py (riscritto)
- **BC warm-start:** models/bc_v2.pth
- **Spazio osservazione:** 8 feature, z-score normalizzato
- **Step allenati:** 100,488 (37 sessioni)
- **Durata:** ~7.3 minuti
- **Modello:** models/rl_bc_warmstart_v3_fixed/final.zip
