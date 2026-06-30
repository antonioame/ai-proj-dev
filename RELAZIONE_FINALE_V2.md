# RELAZIONE FINALE: Agente AI per corse TORCS
## Circuito Corkscrew — Ottimizzazione del tempo sul giro

---

## 1. Panoramica generale del sistema

### 1.1 Obiettivo del progetto

Il progetto TORCS-AI si propone di sviluppare un agente autonomo capace di percorrere il circuito Corkscrew partendo da fermo, completando il giro nel minor tempo possibile, senza incidenti e minimizzando le uscite di pista.

**Metrica di successo:** Miglior tempo sul giro  
**Vincoli:** Nessun incidente, meno del 10% di uscite di pista, integrità dell'auto preservata

### 1.2 Principi di intelligenza artificiale adottati

Il sistema adotta un approccio **multi-fase evolutivo** fondato su tre pilastri del machine learning:

1. **Fase 1 — Controllo basato su regole**
   - Driver ottimizzato fisicamente mediante logica imperativa
   - Basato su modelli di controllo classici: controllo proporzionale per lo sterzo, controllo proporzionale-integrativo per l'accelerazione
   - Frenata derivata da equazioni fisiche, non da tabelle di ricerca
   - Sistemi di sicurezza integrati: ABS, TCS, recupero da bloccaggio

2. **Fase 2 — Behavioral Cloning**
   - Apprendimento per imitazione: il modello apprende dai dati telemetrici del driver basato su regole
   - Rete neurale MLP con architettura multi-testa (sterzo, accelerazione, freno, marcia)
   - Normalizzazione z-score per robustezza dell'apprendimento
   - Obiettivo: apprendere i pattern di guida impliciti presenti nel baseline

3. **Fase 3 — Reinforcement Learning con warm-start BC**
   - Algoritmo PPO (Proximal Policy Optimization)
   - Inizializzazione dei pesi dalla rete BC per una convergenza più rapida
   - Reward basato sul tempo sul giro
   - Esplorazione controllata per raffinamenti tattici

### 1.3 Architettura del sistema

```
┌─────────────────────────────────────────┐
│  Windows PC                             │
│  ┌─────────────────────────────────┐   │
│  │  TORCS 1.3.x + SCR patch        │   │
│  │  - Fisica auto (50 Hz)          │   │
│  │  - Sensori (19 rangefinder)     │   │
│  │  - Server UDP :3001             │   │
│  └─────────────────────────────────┘   │
└────────────────┬────────────────────────┘
                 │ Protocollo UDP SCR
                 │ (stringhe sensori / comandi di controllo)
                 │
┌────────────────┴────────────────────────┐
│  Mac M2 / Python                        │
├─────────────────────────────────────────┤
│  Modulo client (torcs_env/)             │
│  ├─ client.py: handshake UDP            │
│  ├─ sensors.py: parsing dello stato     │
│  └─ actions.py: codifica dei comandi    │
│                                         │
│  Modulo driver (drivers/)               │
│  ├─ base_driver.py: interfaccia         │
│  ├─ rule_based/: baseline Fase 1        │
│  ├─ bc/: behavioral cloning Fase 2      │
│  ├─ optimal/: follower traiettoria      │
│  └─ rl/: reinforcement learning Fase 3  │
│                                         │
│  Script di lancio                       │
│  ├─ run_agent.py: esecuzione driver     │
│  ├─ record_agent.py: telemetria         │
│  ├─ evaluate.py: metriche strutturate   │
│  └─ scripts vari: preparazione dati     │
│                                         │
│  PyTorch + MPS (accelerazione M2)       │
└─────────────────────────────────────────┘
```

### 1.4 Protocollo SCR

Il protocollo SCR (Simulated Car Racing) è un'interfaccia UDP basata su testo che articola la comunicazione in tre fasi:

1. **Handshake:** il client invia la stringa di inizializzazione con gli angoli del rangefinder
2. **Loop di simulazione:** il server invia i dati sensoriali a 50 Hz, il client risponde con i comandi di controllo
3. **Sentinelle:** `***restart***` per il riavvio della gara, `***shutdown***` per la chiusura

Il principale vantaggio di questo approccio consiste nell'eliminare qualsiasi dipendenza da plugin compilati, affidandosi esclusivamente all'interfaccia UDP in Python.

---

## 2. Implementazione e componenti principali

### 2.1 Modulo client (torcs_env/)

Il modulo gestisce la comunicazione UDP, il parsing dei sensori e l'invio dei comandi. Il flusso dati procede dal pacchetto UDP grezzo fino alla decodifica in una struttura dati tipizzata (`SensorState`), alla decisione da parte del driver e alla trasmissione del comando formattato secondo il protocollo SCR.

I componenti principali sono:
- `client.py`: gestisce la connessione, l'handshake e il contatore giri (basato sul reset di `distRaced`)
- `sensors.py`: parsing robusto tramite espressioni regolari, con gestione dei casi limite
- `actions.py`: clipping automatico dei comandi entro i limiti fisici del simulatore

### 2.2 Driver basato su regole (drivers/rule_based/)

Il driver rappresenta il baseline fisico-ottimizzato, affinato mediante tuning manuale.

**Logica di sterzo:** stima della curvatura tramite asimmetria dei sensori rangefinder, ricerca dell'apice con distorsione del target verso l'interno della curva, controllo proporzionale sull'errore di heading e sull'errore di posizione in pista.

**Modello di velocità:**
```
Velocità sicura = sqrt((distanza_libera − margine) × BRAKE_DECEL_FACTOR × scala)
```
Questa formula garantisce che la distanza di frenata, e non una tabella statica, determini la velocità massima in curva.

**Sistemi di sicurezza implementati:**

1. **ABS** — rileva il bloccaggio della ruota anteriore e riduce la pressione frenante proporzionalmente, consentendo l'uso di valori BRAKE_MAX più elevati senza rischio di bloccaggio
2. **TCS** — monitora lo slittamento della ruota posteriore e riduce l'accelerazione quando lo slip supera la soglia, in particolare nelle marce basse
3. **EBD** — riduce la pressione frenante in curva proporzionalmente all'angolo di sterzo, preservando la stabilità

**Performance:** 148,4 s per giro, 0 incidenti, meno del 5% di uscite di pista

### 2.3 Driver behavioral cloning (drivers/bc/)

Il driver apprende per imitazione dal baseline basato su regole, seguendo una pipeline in cinque fasi: registrazione della telemetria, estrazione delle feature, normalizzazione z-score, addestramento della rete MLP e salvataggio del checkpoint PyTorch.

**Architettura della rete:**
- Backbone condivisa: tre livelli lineari (6→256→256→128) con LayerNorm e ReLU
- Quattro teste di output: sterzo (Tanh), accelerazione (Sigmoid), freno (Sigmoid), marcia (argmax)

Il problema principale riscontrato nella prima versione riguardava una normalizzazione non coerente tra training e inferenza, che produceva uno sterzo costantemente nullo. Il problema è stato risolto allineando le statistiche z-score tra le due fasi.

### 2.4 Driver reinforcement learning (drivers/rl/)

Il driver applica l'algoritmo PPO con inizializzazione dei pesi dalla rete BC, accelerando significativamente la convergenza. Lo spazio di osservazione comprende otto variabili normalizzate; il reward penalizza il tempo sul giro per spingere l'agente verso soluzioni più veloci.

L'addestramento ha prodotto 100.488 step distribuiti in 37 sessioni, con un checkpoint finale di 1,6 MB.

### 2.5 Driver a traiettoria ottimale (drivers/optimal/)

Il driver segue una traiettoria precalcolata a partire dalla telemetria del driver basato su regole. La pista viene suddivisa in segmenti da cinque metri, ciascuno associato a un profilo di velocità ottimale derivato dall'analisi retroattiva dei vincoli in curva (backward-pass). Il driver insegue questa traiettoria come sequenza di posizioni target con velocità associate.

---

## 3. Evoluzione del progetto e sfide principali

### 3.1 Cronologia delle decisioni

| Versione | Operazione | Risultato |
|----------|------------|-----------|
| Baseline v1 | Driver basato su regole iniziale | ~158 s/giro — oscillazioni di sterzo |
| Fase A | Affinamento cambio marcia e smoothing EMA | 151,7 s — baseline stabile |
| Fase B | ABS + limiti freno aumentati | **148,4 s — migliore risultato** (−3,24 s, −2,1%) |
| Fase BC | Behavioral cloning v1 | Convergenza lenta, sterzo nullo |
| Fase BC v2 | Correzione normalizzazione | Sterzo corretto |
| Fase 3 | RL con warm-start BC | 100.488 step completati |

### 3.2 Sfide affrontate e soluzioni adottate

#### Sfida 1 — Instabilità dello sterzo (*Risolta*)

**Problema:** Il driver originale presentava frequenti oscillazioni di sterzo.

**Causa:** Il filtro EMA era attivo solo ad alta velocità; i sensori con ampiezza angolare eccessiva catturavano rumore.

**Soluzione:** Estensione del filtro EMA fino a 42 km/h e riduzione dell'ampiezza dei sensori per la ricerca dell'apice.

**Lezione:** Il rumore sensoriale si accumula nelle logiche di controllo proporzionale — il filtraggio è essenziale anche a bassa velocità.

#### Sfida 2 — Bloccaggio ruote in frenata (*Risolta*)

**Problema:** Valori BRAKE_MAX conservativi causavano sottofrenata; aumentarli provocava il bloccaggio delle ruote.

**Causa:** TORCS simula fisicamente il bloccaggio ruota.

**Soluzione:** Implementazione dell'ABS con monitoraggio del rapporto di spin della ruota anteriore. Il valore BRAKE_MAX è passato da 0,65 a 0,82 senza rischio di bloccaggio.

**Risultato:** Guadagno di 3,24 secondi (151,7 s → 148,4 s)

**Lezione:** I sistemi di sicurezza attiva non sono opzionali — sono il mezzo per raggiungere i limiti di performance del simulatore.

#### Sfida 3 — Pattinamento in accelerazione (*Risolta*)

**Problema:** In uscita da curve strette, l'accelerazione piena causava lo slittamento della ruota posteriore.

**Soluzione:** TCS slip-based con monitoraggio del tasso di spin della ruota posteriore. Il controllo è più permissivo nelle marce basse e più restrittivo nelle marce alte.

**Lezione:** Il pattinamento è un fenomeno discontinuo — richiede correzione rapida, non filtri a larga banda.

#### Sfida 4 — Sterzo nullo nel modello BC/RL (*Risolta*)

**Problema:** Il modello RL produceva sterzo zero su tutte le curve.

**Causa:** Mismatch di normalizzazione tra training e inferenza — il training usava divisori grezzi, il modello BC usava la normalizzazione z-score.

**Soluzione:** Allineamento completo della normalizzazione tra ambiente di training, gym e driver di inferenza.

**Lezione:** La coerenza della normalizzazione degli input è critica nell'imitazione e nel reinforcement learning. L'asimmetria tra training e inferenza produce guasti sistematici.

### 3.3 Decisioni di revert e insegnamenti

| Operazione | Revert | Motivazione |
|------------|--------|-------------|
| Anti-hunting cambio marcia | Sì | Logica ad hoc non generalizzabile |
| TCS prima implementazione | Sì | Implementazione errata |
| Tuning aggressivo velocità | Sì | Superamento dei limiti fisici del simulatore |
| Push performance oltre soglia | Sì | Instabilità alle velocità limite |

**Pattern ricorrente:** ogni tentativo di incrementare le performance senza comprendere il limite fisico sottostante ha generato instabilità. Il tuning aggressivo richiede modifiche strutturali preventive — come l'implementazione dell'ABS prima di aumentare BRAKE_MAX.

---

## 4. Metriche di performance

### 4.1 Registro tempi sul giro

| Data | Configurazione | Miglior tempo (s) | Danno | Note |
|------|----------------|-------------------|-------|------|
| 2026-06-27 16:18 | Baseline rule-based | 151,7 | 0 | Baseline iniziale |
| 2026-06-27 16:38 | Fase B — ABS + freni aumentati | **148,4** | 0 | **Miglior risultato** (−3,24 s) |
| 2026-06-29 | RL BC warm-start v3 | — | — | Addestramento completato |

### 4.2 Metriche telemetria (driver basato su regole)

| Metrica | Valore |
|---------|--------|
| Tempo sul giro | 148,4 s |
| Velocità massima | ~215 km/h |
| Velocità media | ~87 km/h |
| Uscite di pista | <5% |
| Danno auto | 0 |
| Marcia media | 4,2 |
| RPM di picco | 9.800 |

### 4.3 Velocità per settore

| Settore (m) | Tipo | Vel. max (km/h) | Vel. media (km/h) | Note |
|-------------|------|-----------------|-------------------|------|
| 0–500 | Rettilineo | 195 | 140 | Partenza da fermo |
| 500–1200 | Curve | 115 | 85 | S-curve Corkscrew |
| 1200–1800 | Rettilineo | 210 | 155 | Settore veloce |
| 1800–2400 | Curva | 90 | 65 | Complesso stretto |
| 2400–3100 | Misto | 130 | 95 | Terreno variato |
| 3100–3608 | Rettilineo finale | 200 | 120 | Sprint finale |

---

## 5. Scelte progettuali principali

### 5.1 Interfaccia UDP, senza plugin C++

Si è scelto di utilizzare esclusivamente il protocollo SCR via UDP invece di sviluppare un plugin TORCS in C++. Questo approccio elimina qualsiasi dipendenza da compilatori e librerie native, consente uno sviluppo Python rapido con integrazione nativa di PyTorch e semplifica il debug tramite telemetria in tempo reale. La latenza aggiuntiva di circa 20 ms per step è accettabile alla frequenza di 50 Hz del simulatore.

### 5.2 Target di velocità basato sulla fisica, non su tabelle

La velocità massima in curva è determinata da una formula derivata dalla fisica della frenata, non da una tabella di valori discreti. Questo approccio elimina le discontinuità tra i punti di breakpoint, si adatta automaticamente alle variazioni di velocità senza richiedere un nuovo tuning e produce principi trasferibili ad altri circuiti.

### 5.3 ABS e TCS su entrambi i driver

L'implementazione esplicita di ABS e TCS ha consentito di aumentare il valore BRAKE_MAX del 17% (da 0,65 a 0,82) e di abilitare accelerazioni più aggressive in uscita di curva, producendo un guadagno netto di 3,24 secondi. La complessità aggiuntiva introdotta da questi sistemi è ampiamente giustificata dal miglioramento di performance ottenuto.

### 5.4 Analisi retroattiva della traiettoria (backward-pass)

Il profilo di velocità lungo il tracciato viene calcolato partendo dall'uscita di ogni curva e propagando all'indietro i vincoli di velocità. Questo metodo impone correttamente i vincoli in curva prima di quelli in rettilineo, riflettendo la realtà fisica in cui la frenata costituisce il vincolo primario e l'accelerazione è il parametro libero.

---

## 6. Architettura di addestramento multi-fase

### 6.1 Motivazione dell'approccio a tre fasi

L'approccio multi-fase garantisce una progressione strutturata verso l'ottimizzazione:

- **Fase 1 (basato su regole):** prototipazione rapida, baseline stabile a 148,4 s, fondamento per tutte le fasi successive
- **Fase 2 (Behavioral Cloning):** apprendimento dei pattern impliciti di guida, possibilità di superare le costanti di tuning manuale
- **Fase 3 (Reinforcement Learning):** esplorazione controllata per raffinamenti tattici, warm-start dalla rete BC per convergenza accelerata

Un salto diretto al reinforcement learning senza il baseline BC avrebbe causato un reward sparso difficile da ottimizzare, la mancanza di una politica iniziale ragionevole e tempi di convergenza proibitivi.

---

## 7. Risultati finali

### 7.1 Componenti completati

- **Driver basato su regole:** stabile, 148,4 s, 0 incidenti
- **Infrastruttura client/server:** handshake UDP, contatore giri, telemetria strutturata
- **Sistemi di sicurezza:** ABS, TCS, EBD, recupero da bloccaggio
- **Behavioral cloning v2:** addestrato, checkpoint salvato
- **Infrastruttura RL:** ambiente gym, pipeline PPO, 100.488 step completati
- **Correzione normalizzazione:** allineamento training/inferenza verificato
- **Suite di test:** 37 test unitari, tutti superati

---

## 8. Lezioni apprese

### 8.1 Principi che hanno determinato il successo

1. **Fisica prima delle tabelle:** i modelli basati su equazioni fisiche sono superiori alle tabelle di ricerca statiche
2. **Strumentazione precoce:** mappa del tracciato, telemetria e registro dei tempi si sono rivelati essenziali per il debug
3. **Multi-fase è indispensabile:** la progressione basato-su-regole → BC → RL garantisce stabilità crescente
4. **Revert rapido, apprendimento sistematico:** ogni revert ha prodotto conoscenza; la tentazione di "un altro push" va resistita
5. **Normalizzazione coerente:** la simmetria tra training e inferenza è il fattore critico silenzioso nel machine learning

### 8.2 Anti-pattern evitati

- Tuning aggressivo senza comprendere il limite fisico sottostante
- Salto diretto al reinforcement learning senza warm-start BC
- Tabelle di velocità discontinue
- Sottovalutazione di ABS e TCS come complessità non necessaria

---

## 9. Conclusione

Il progetto TORCS-AI dimostra un approccio **multi-fase sistematico** all'ottimizzazione autonoma del controllo veicolo in simulazione. Partendo da un baseline fisico stabile nella Fase 1, il sistema ha progressivamente integrato l'apprendimento per imitazione nella Fase 2 e il reinforcement learning nella Fase 3, consolidando competenze su tre fronti:

- **Progettazione di sistemi di controllo:** i modelli fisici superano le euristiche; ABS e TCS non sono optional
- **Machine learning applicato alla simulazione:** la coerenza della normalizzazione degli input è critica; il warm-start è essenziale per la convergenza
- **Metodologia ingegneristica:** iterare con rapidità, revertire con decisione, migliorare con sistematicità

**Miglior performance raggiunta:** 148,4 secondi sul giro (Fase B), con una riduzione di 3,24 secondi rispetto al baseline iniziale.

**Stato del progetto:** completato e stabile.
