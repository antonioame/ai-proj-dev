# RELAZIONE FINALE: Agente AI per corse TORCS
## Circuito Corkscrew — Ottimizzazione del tempo sul giro

---

## 1. Panoramica generale del sistema

### 1.1 Obiettivo del progetto

Il progetto TORCS-AI si propone di sviluppare un agente autonomo capace di percorrere il circuito Corkscrew partendo da fermo, completando il giro nel minor tempo possibile, senza incidenti e minimizzando le uscite di pista.

**Metrica di successo:** Miglior tempo sul giro  
**Vincoli:** Nessun incidente, escursioni fuori pista minime, integrità dell'auto preservata

### 1.2 Principi di intelligenza artificiale adottati

Il sistema adotta un approccio **multi-fase evolutivo** fondato su tre pilastri del machine learning:

1. **Fase 1 — Controllo basato su regole**
   - Driver ottimizzato fisicamente mediante logica imperativa
   - Basato su modelli di controllo classici: controllo proporzionale per lo sterzo, controllo proporzionale-integrativo per l'accelerazione
   - Frenata derivata da equazioni fisiche, non da tabelle di ricerca
   - Sistemi di sicurezza integrati: ABS, TCS, EBD, recupero da bloccaggio
   - Oggi isolato come riferimento storico, non più il driver primario

2. **Fase 2 — Behavioral Cloning ibrido**
   - Apprendimento per imitazione da **due sorgenti distinte**, fuse dinamicamente in base al contesto di pista: un sotto-modello per i rettilinei, addestrato sulla telemetria di un precedente tentativo di driving-net (a sua volta addestrato su dati del driver rule-based); un sotto-modello per le curve, addestrato sulla telemetria di un vecchio driver personale ibrido (regole + predittore BC)
   - Rete neurale MLP con quattro teste di output (sterzo, accelerazione, freno, marcia)
   - Normalizzazione z-score per robustezza dell'apprendimento
   - Guadagni post-hoc e cambio marcia basato su RPM applicati fuori dalla rete
   - **Driver candidato alla consegna finale**

3. **Fase 3 — Reinforcement Learning con warm-start BC**
   - Algoritmo **SAC (Soft Actor-Critic)**, Stable-Baselines3
   - Inizializzazione dei pesi dell'attore dal solo sotto-modello BC per le curve (non esiste un'unica rete BC da cui partire, essendo BC un blend di due reti)
   - Due varianti testate: SAC diretto (sostituisce interamente il controllo) e **residual** (correzione limitata sopra il driver BC completo)
   - Solo la variante residual completa il giro in sicurezza; non ancora promossa a driver primario perché più lenta di BC

### 1.3 Architettura del sistema

```
┌─────────────────────────────────────────┐
│  Windows PC                             │
│  ┌─────────────────────────────────┐   │
│  │  TORCS 1.3.x + patch SCR         │   │
│  │  - Fisica auto (50 Hz)          │   │
│  │  - Sensori (19 rangefinder)     │   │
│  │  - Server UDP :3001             │   │
│  └─────────────────────────────────┘   │
└────────────────┬────────────────────────┘
                 │ Protocollo UDP SCR
                 │ (stringhe sensori / comandi di controllo)
                 │
┌────────────────┴────────────────────────┐
│  Mac M2 / Python (client + training)    │
├─────────────────────────────────────────┤
│  Modulo protocollo (torcs_env/)         │
│  ├─ client.py: handshake UDP, contatore │
│  │  giri                                │
│  ├─ sensors.py: parsing dello stato     │
│  └─ actions.py: codifica dei comandi    │
│                                         │
│  Driver in primo piano (_DRIVER/)       │
│  └─ driver.py: BCDriver, blend di due   │
│     modelli (rettilineo/curva) + guadagni│
│                                         │
│  Driver isolato di riferimento          │
│  └─ old_versions_drivers/project_V2/:   │
│     rule_based (~148 s, non collegato   │
│     agli script principali)             │
│                                         │
│  Driver RL — Fase 3 (drivers/rl/)       │
│  ├─ driver.py: RLDriver (SAC diretto,   │
│  │  non funzionante da solo)            │
│  └─ residual_driver.py: ResidualRLDriver│
│     (base BC + correzione SAC, funziona)│
│                                         │
│  Infrastruttura training RL             │
│  (training/rl/): wrapper Gymnasium,     │
│  reward versionato, warm-start SAC      │
│                                         │
│  Script di lancio (scripts/)            │
│  ├─ run_agent.py / run_agent_rl.py      │
│  ├─ evaluate.py / evaluate_rl.py        │
│  └─ record_agent.py, benchmark.py, ...  │
│                                         │
│  PyTorch (+ backend MPS su Mac M2)      │
└─────────────────────────────────────────┘
```

### 1.4 Protocollo SCR

Il protocollo SCR (Simulated Car Racing) è un'interfaccia UDP basata su testo che articola la comunicazione in tre fasi:

1. **Handshake:** il client invia la stringa di inizializzazione con gli angoli del rangefinder
2. **Loop di simulazione:** il server invia i dati sensoriali a 50 Hz (~20 ms per step), il client risponde con i comandi di controllo
3. **Sentinelle:** `***restart***` per il riavvio della gara, `***shutdown***` per la chiusura

Il principale vantaggio di questo approccio consiste nell'eliminare qualsiasi dipendenza da plugin compilati, affidandosi esclusivamente all'interfaccia UDP in Python.

---

## 2. Implementazione e componenti principali

### 2.1 Modulo protocollo (torcs_env/)

Il modulo gestisce la comunicazione UDP, il parsing dei sensori e l'invio dei comandi. Il flusso dati procede dal pacchetto UDP grezzo fino alla decodifica in una struttura dati tipizzata (`SensorState`), alla decisione da parte del driver e alla trasmissione del comando formattato secondo il protocollo SCR.

I componenti principali sono:
- `client.py`: gestisce la connessione, l'handshake e il contatore giri (basato sul reset di `distRaced`)
- `sensors.py`: parsing robusto tramite espressioni regolari, con gestione dei casi limite
- `actions.py`: clipping automatico dei comandi entro i limiti fisici del simulatore

### 2.2 Driver basato su regole — isolato, di solo riferimento (old_versions_drivers/project_V2/)

Questo driver rappresenta il baseline fisico-ottimizzato della Fase 1, affinato mediante tuning manuale. È oggi **isolato**: non più collegato agli script principali (`scripts/run_agent.py`, `scripts/evaluate.py`), superato in performance dal driver BC ibrido e conservato solo come riferimento storico.

**Logica di sterzo:** stima della curvatura tramite asimmetria dei sensori rangefinder, ricerca dell'apice con distorsione del target verso l'interno della curva, controllo proporzionale sull'errore di heading e sull'errore di posizione in pista.

**Modello di velocità:**
```
Velocità sicura = sqrt((distanza_libera − margine) × BRAKE_DECEL_FACTOR × scala)
```
Questa formula garantisce che la distanza di frenata, e non una tabella statica, determini la velocità massima in curva.

**Sistemi di sicurezza implementati:**

1. **ABS** — rileva il bloccaggio della ruota anteriore e riduce la pressione frenante proporzionalmente, consentendo l'uso di valori BRAKE_MAX più elevati senza rischio di bloccaggio
2. **TCS** — monitora lo slittamento della ruota posteriore e riduce l'accelerazione quando lo slip supera la soglia, con un guadagno di correzione più aggressivo nelle marce basse e più permissivo nelle marce alte
3. **EBD** — riduce la pressione frenante in curva proporzionalmente all'angolo di sterzo, preservando la stabilità

**Performance:** 148,4 s per giro, 0 incidenti, meno del 5% di uscite di pista

### 2.3 Driver Behavioral Cloning ibrido — candidato alla consegna (_DRIVER/)

A differenza di un singolo modello addestrato per imitazione, il driver in primo piano è un **blend di due reti separate**, selezionate dinamicamente in base al contesto di pista:

- **Sotto-modello rettilineo** (`bc_from_attempt1_v1`): addestrato sulla telemetria registrata facendo guidare un precedente tentativo di driving-net (`_DRIVER/bc_source_driver/`), a sua volta addestrato su dati del driver rule-based
- **Sotto-modello curva** (`bc_from_olddriver_v1`): addestrato sulla telemetria di un vecchio driver personale ibrido (regole + predittore BC), il più generalista dei due e per questo scelto anche come base di warm-start per la Fase 3

Il peso di fusione è determinato dalla distanza del sensore frontale (`track[9]`): oltre 44 m si usa il modello rettilineo puro, sotto i 22 m il modello curva puro, con transizione lineare morbida nella zona intermedia.

**Architettura della rete (per ciascun sotto-modello):**
- Backbone condivisa: due livelli lineari (26→128→64) con ReLU, dove 26 è la dimensione del vettore di feature (angle, speed, speedY, speedZ, trackPos, 19 rangefinder, rpm, gear)
- Quattro teste di output: sterzo (Tanh), accelerazione (Sigmoid), freno (Sigmoid), marcia (regressione lineare, non usata in produzione — il cambio marcia effettivo è gestito da una logica RPM esterna alla rete)
- Normalizzazione z-score allineata tra training e inferenza

**Guadagni post-hoc** applicati all'uscita fusa (STEER_GAIN 1,8 / ACCEL_GAIN 1,40 / BRAKE_GAIN 0,80) e cambio marcia automatico basato su soglie RPM (salita oltre 12.000 rpm, discesa sotto 6.000 rpm) completano la pipeline.

**Performance:** **121,978 s**, 199,6 km/h di punta, 0% di uscite di pista — miglior risultato del progetto, ottenuto restringendo le soglie di blend rettilineo/curva (120→44 m e 60→22 m) rispetto alla prima versione del blend (125,790 s).

### 2.4 Driver Reinforcement Learning — Fase 3 (drivers/rl/, training/rl/)

Il driver applica l'algoritmo **SAC (Soft Actor-Critic)**, scelto per l'efficienza campionaria off-policy e la stabilità di training superiore a DDPG. Lo spazio di osservazione comprende le stesse **26 feature normalizzate** usate da BC (condivise tramite un'unica funzione, `training/rl/features.py`, per evitare il disallineamento che aveva afflitto un precedente tentativo RL — poi rimosso). Lo spazio d'azione è limitato a sterzo/accelerazione/freno; la marcia resta automatica.

Sono state esplorate due varianti:

1. **SAC diretto**, con warm-start dei pesi dal solo sotto-modello BC per le curve: con qualunque versione del reward, la policy sfrutta l'intera autorità di controllo per massimizzare la velocità istantanea a scapito della guida — un caso di *reward hacking* che porta l'auto a bloccarsi (0 giri completati, velocità media inferiore a 1 km/h).
2. **RL residual** (approccio adottato): la rete SAC non sostituisce il driver BC ma apprende una **correzione limitata** sopra di esso — `azione_finale = BCDriver.step(stato) + 0,03 × correzione_SAC` — con una penalità L2 che tiene la correzione vicina allo zero. All'inizio del training l'agente guida esattamente come BC e completa giri da subito; il training affina poi piccoli aggiustamenti dipendenti dallo stato.

**Performance del driver residual:** 127,07 s, 0% di uscite di pista, 0 danni — completa il giro in sicurezza ma è circa il 4% più lento di BC. Per questo motivo **non è stato promosso** a driver primario: resta il driver RL dimostrativo, genuinamente funzionante.

Il reward per-step (formula base del corso: `v·cos(angle) − v·|sin(angle)| − v·|trackPos|`) è stato affiancato da una seconda versione raffinata empiricamente, con un termine di progresso proporzionale al `distRaced` percorso, una penalità per l'auto ferma e una penalità di uscita pista raddoppiata rispetto alla formula base.

### 2.5 Tentativo di driver a traiettoria ottimale — abbandonato

Un ulteriore approccio, basato su una traiettoria precalcolata a partire dalla telemetria del driver rule-based (segmenti di pista da cinque metri, ciascuno con un profilo di velocità ottimale derivato da un'analisi retroattiva dei vincoli in curva), è stato implementato e successivamente **rimosso** perché non funzionante in pista. Il principio di analisi retroattiva (backward-pass) resta documentato come scelta progettuale esplorata (vedi §5.4) ma non è presente in nessuno dei driver attualmente attivi.

---

## 3. Evoluzione del progetto e sfide principali

### 3.1 Cronologia delle decisioni

| Versione | Operazione | Risultato |
|----------|------------|-----------|
| Baseline v1 | Driver basato su regole iniziale | ~158 s/giro — oscillazioni di sterzo |
| Fase A | Affinamento cambio marcia e smoothing EMA | 151,7 s — baseline stabile |
| Fase B | ABS + limiti freno aumentati | 148,4 s — driver rule-based isolato di riferimento |
| Fase BC | Behavioral cloning ibrido (attempt1 + olddriver) | 125,790 s |
| Fase BC tuning | Soglie di blend rettilineo/curva ristrette (120→44 m, 60→22 m) | **121,978 s — migliore risultato attuale** |
| Fase BC self-distill | Modello singolo per distillazione (non adottato) | Schiantato fuori pista, mai completato un giro |
| Fase 3 RL diretto | SAC warm-start puro, senza base BC | Reward hacking — auto bloccata, 0 giri |
| Fase 3 RL residual | Base BC + correzione SAC limitata | 127,07 s, 0% fuori pista — funzionante ma non promosso |

### 3.2 Sfide affrontate e soluzioni adottate

#### Sfida 1 — Instabilità dello sterzo (*Risolta*)

**Problema:** Il driver originale presentava frequenti oscillazioni di sterzo.

**Causa:** Il filtro EMA era attivo solo ad alta velocità; i sensori con ampiezza angolare eccessiva catturavano rumore.

**Soluzione:** Estensione del filtro EMA fino a 42 km/h e riduzione dell'ampiezza dei sensori per la ricerca dell'apice.

**Lezione:** Il rumore sensoriale si accumula nelle logiche di controllo proporzionale — il filtraggio è essenziale anche a bassa velocità.

#### Sfida 2 — Bloccaggio ruote in frenata (*Risolta*)

**Problema:** Valori BRAKE_MAX conservativi causavano sottofrenata; aumentarli provocava il bloccaggio delle ruote.

**Causa:** TORCS simula fisicamente il bloccaggio ruota.

**Soluzione:** Implementazione dell'ABS con monitoraggio del rapporto di spin della ruota anteriore. Il valore BRAKE_MAX (per il regime di velocità più alto) è passato da 0,65 a 0,82 senza rischio di bloccaggio (+26% circa).

**Risultato:** Guadagno di 3,24 secondi (151,7 s → 148,4 s)

**Lezione:** I sistemi di sicurezza attiva non sono opzionali — sono il mezzo per raggiungere i limiti di performance del simulatore.

#### Sfida 3 — Pattinamento in accelerazione (*Risolta*)

**Problema:** In uscita da curve strette, l'accelerazione piena causava lo slittamento della ruota posteriore.

**Soluzione:** TCS slip-based con monitoraggio del tasso di spin della ruota posteriore. Il controllo è più restrittivo nelle marce basse (dove il pattinamento è più probabile) e più permissivo nelle marce alte.

**Lezione:** Il pattinamento è un fenomeno discontinuo — richiede correzione rapida, non filtri a larga banda.

#### Sfida 4 — Latenza per-step nel training RL (*Risolta*)

**Problema:** Ogni tentativo di training RL falliva silenziosamente: gli episodi finivano fuori pista dopo circa 300 step, indipendentemente dalla policy in uso — verificato forzando l'azione a puro BC, che si schiantava comunque.

**Causa:** TORCS in modalità headless (`-r`) gira sul proprio clock e non aspetta un client lento: continua ad avanzare la simulazione con l'ultima azione ricevuta. L'update del gradiente di default di Stable-Baselines3 (eseguito dopo ogni singolo step, ~10-30 ms su CPU) introduceva un ritardo sufficiente a far derivare l'auto dalla traiettoria durante il lancio ad alta velocità.

**Soluzione:** Due correzioni congiunte — (1) training per-episodio (`train_freq=(1, "episode")`), così gli update del gradiente avvengono tra un episodio e l'altro, ad auto ferma, non durante la guida; (2) lancio di TORCS differito al primo step dell'episodio, invece che nel `reset()`, così nessun processo TORCS resta in attesa durante il blocco di update del gradiente eseguito da SB3 tra `reset()` e il primo `step()`.

**Lezione:** In un ambiente RL basato su un simulatore in tempo reale che non attende il client, la latenza di training va trattata come un vincolo di sistema, non solo come un problema di velocità — un ritardo anche piccolo può corrompere silenziosamente ogni run, mascherandosi da problema di policy.

### 3.3 Decisioni di revert e insegnamenti

| Operazione | Revert | Motivazione |
|------------|--------|-------------|
| Anti-hunting cambio marcia | Sì | Logica ad hoc non generalizzabile |
| TCS prima implementazione | Sì | Implementazione errata |
| Tuning aggressivo velocità | Sì | Superamento dei limiti fisici del simulatore |
| Push performance oltre soglia | Sì | Instabilità alle velocità limite |
| Soglie di blend BC oltre 44/22 m | Sì | Comportamento fragile e non monotono (133 s) |
| Self-distillation BC (modello singolo) | Sì | Nessun esempio di recupero da errore nei dati, auto schiantata fuori pista |
| Tuning manuale dei guadagni STEER/ACCEL/BRAKE di BC | Sì | Ogni tentativo, in entrambe le direzioni, ha peggiorato il tempo o l'uscita di pista |

**Pattern ricorrente:** ogni tentativo di incrementare le performance senza comprendere il limite fisico o statistico sottostante ha generato instabilità. Il tuning aggressivo richiede modifiche strutturali preventive — come l'implementazione dell'ABS prima di aumentare BRAKE_MAX, o dati di recupero da errore prima di affidarsi a un singolo modello distillato.

---

## 4. Metriche di performance

### 4.1 Registro tempi sul giro

| Data / commit | Configurazione | Miglior tempo (s) | Danno | Note |
|------|----------------|-------------------|-------|------|
| 2026-06-27 16:18 | Baseline rule-based | 151,7 | 0 | Baseline iniziale |
| 2026-06-27 16:38 | Fase B — ABS + freni aumentati | 148,4 | 0 | Driver rule-based, oggi isolato di riferimento |
| bcfe1f9 | BC ibrido attempt1 + olddriver | 125,790 | 0 | Prima versione del blend BC |
| 24ab766 | BC ibrido, soglie di blend ristrette | **121,978** | 0 | **Miglior risultato attuale — driver candidato alla consegna** |
| — | RL diretto (SAC warm-start, senza base BC) | — | — | Reward hacking, 0 giri completati |
| 2026-07-10 | RL residual (base BC + correzione SAC) | 127,070 | 0 | Driver RL funzionante, ~4% più lento di BC — non promosso |

### 4.2 Metriche telemetria (driver rule-based, isolato di riferimento)

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

La velocità massima in curva è determinata da una formula derivata dalla fisica della frenata, non da una tabella di valori discreti. Questo approccio elimina le discontinuità tra i punti di breakpoint, si adatta automaticamente alle variazioni di velocità senza richiedere un nuovo tuning e produce principi trasferibili ad altri circuiti. È il principio alla base del driver rule-based.

### 5.3 ABS e TCS sul driver rule-based

L'implementazione esplicita di ABS e TCS ha consentito di aumentare il valore BRAKE_MAX di punta di circa il 26% (da 0,65 a 0,82) e di abilitare accelerazioni più aggressive in uscita di curva, producendo un guadagno netto di 3,24 secondi sul driver rule-based. La complessità aggiuntiva introdotta da questi sistemi è ampiamente giustificata dal miglioramento di performance ottenuto. Il driver BC, che lo ha superato, non replica questa logica esplicita: la sicurezza emerge invece dai pattern di guida imitati e dai guadagni post-hoc applicati all'uscita del blend.

### 5.4 Analisi retroattiva della traiettoria (backward-pass) — principio esplorato, non in produzione

Il principio prevede di calcolare il profilo di velocità lungo il tracciato partendo dall'uscita di ogni curva e propagando all'indietro i vincoli di velocità, imponendo correttamente i vincoli in curva prima di quelli in rettilineo. È il principio su cui si basava il driver a traiettoria ottimale (§2.5), poi rimosso perché non funzionante in pista: nessuno dei driver oggi attivi (rule-based, BC, RL) usa una traiettoria precalcolata — tutti operano in modo reattivo, sensore per sensore, step dopo step.

---

## 6. Architettura di addestramento multi-fase

### 6.1 Motivazione dell'approccio a tre fasi

L'approccio multi-fase garantisce una progressione strutturata verso l'ottimizzazione:

- **Fase 1 (basato su regole):** prototipazione rapida, baseline stabile a 148,4 s, fondamento per generare i dati di training della Fase 2
- **Fase 2 (Behavioral Cloning):** apprendimento dei pattern impliciti di guida da due sorgenti distinte, superamento del baseline rule-based (121,978 s) — **driver candidato alla consegna**
- **Fase 3 (Reinforcement Learning):** warm-start dalla rete BC per convergenza accelerata; l'approccio residual mantiene la sicurezza del driver BC aggiungendo una correzione appresa, ma non ne ha ancora superato il tempo sul giro

Un salto diretto al reinforcement learning senza un driver di base già funzionante avrebbe causato un reward sparso difficile da ottimizzare e tempi di convergenza proibitivi. Questa previsione si è confermata empiricamente: il tentativo di SAC diretto, senza la base BC a mantenere l'auto in pista, è caduto in reward hacking e non ha mai completato un giro.

---

## 7. Risultati finali

### 7.1 Componenti completati

- **Driver basato su regole:** stabile, 148,4 s, 0 incidenti — oggi isolato, sostituito dal driver BC come consegna primaria
- **Infrastruttura client/server:** handshake UDP, contatore giri, telemetria strutturata
- **Sistemi di sicurezza del driver rule-based:** ABS, TCS, EBD, recupero da bloccaggio
- **Behavioral cloning ibrido:** **121,978 s, 199,6 km/h, 0% fuori pista — driver candidato alla consegna finale**
- **Infrastruttura RL:** ambiente Gymnasium, algoritmo SAC (diretto e residual), vettore di feature condiviso con BC
- **Driver RL residual:** 127,07 s, 0% fuori pista, 0 danni — funzionante ma non promosso (più lento di BC)
- **Suite di test:** 37 test unitari, tutti superati

---

## 8. Lezioni apprese

### 8.1 Principi che hanno determinato il successo

1. **Fisica prima delle tabelle:** i modelli basati su equazioni fisiche sono superiori alle tabelle di ricerca statiche
2. **Strumentazione precoce:** telemetria e registro dei tempi (`laptime_ledger.csv`) si sono rivelati essenziali per confrontare le configurazioni nel tempo
3. **Multi-fase è indispensabile:** la progressione rule-based → BC → RL residual garantisce sicurezza crescente, pur non migliorando necessariamente il tempo sul giro a ogni fase
4. **Revert rapido, apprendimento sistematico:** ogni revert (soglie di blend, self-distillation, RL diretto) ha prodotto conoscenza riutilizzata nella fase successiva
5. **Normalizzazione e osservazioni coerenti:** condividere un'unica funzione di estrazione feature tra BC e RL ha prevenuto il disallineamento che aveva bloccato un precedente tentativo RL, poi rimosso

### 8.2 Anti-pattern evitati

- Tuning aggressivo senza comprendere il limite fisico sottostante
- Salto diretto al reinforcement learning senza un driver di base funzionante — confermato dal fallimento del tentativo SAC diretto (reward hacking)
- Tabelle di velocità discontinue
- Promozione di un driver più lento solo perché "più moderno" (il residual RL non ha sostituito BC proprio per questo)

---

## 9. Conclusione

Il progetto TORCS-AI dimostra un approccio **multi-fase sistematico** all'ottimizzazione autonoma del controllo veicolo in simulazione. Partendo da un baseline fisico stabile nella Fase 1 (148,4 s, oggi isolato di riferimento), il sistema ha superato quel risultato nella Fase 2 con un driver di behavioral cloning ibrido, e ha esplorato il reinforcement learning nella Fase 3 con un approccio residual che mantiene la sicurezza del driver BC:

- **Progettazione di sistemi di controllo:** i modelli fisici superano le euristiche; ABS e TCS non sono optional per il driver rule-based
- **Machine learning applicato alla simulazione:** la coerenza della normalizzazione e delle feature tra fasi è critica; il warm-start è essenziale per la convergenza, ma non sufficiente da solo a garantire un training RL stabile senza affrontare anche la latenza del loop di training
- **Metodologia ingegneristica:** iterare con rapidità, revertire con decisione, promuovere un driver solo quando eguaglia o supera quello attuale su sicurezza e tempo

**Miglior performance raggiunta:** **121,978 secondi** sul giro (driver Behavioral Cloning ibrido), 199,6 km/h di punta, 0% di uscite di pista — candidato alla consegna finale. Il driver Reinforcement Learning residual (127,07 s) dimostra un approccio RL genuinamente funzionante e sicuro, ma non ancora competitivo sul tempo giro.

**Stato del progetto:** driver BC stabile e pronto alla consegna; driver RL residual funzionante come dimostrazione, non promosso.
