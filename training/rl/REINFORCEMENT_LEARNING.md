# Reinforcement Learning per l'agente di guida TORCS — Base di conoscenza Fase 3

> Destinazione suggerita nel repo: `training/rl/REINFORCEMENT_LEARNING.md`
> (sostituisce il placeholder Fase 3 `README.md` creato durante lo scaffolding del progetto).

## 0. Perché esiste questo documento

Questo progetto ha attualmente **Fase 1 (baseline rule-based)** e **Fase 2 (Behavioral Cloning)**
implementate, addestrate e integrate. Entrambe producono già un agente di guida che completa il
giro del Corkscrew (giro singolo, partenza da fermo, in solitaria, senza avversari, senza
schianti, escursioni fuori pista minime).

Questo documento fornisce a chi implementa la **Fase 3 (fine-tuning con Reinforcement Learning)**
le conoscenze concettuali e pratiche necessarie per farlo correttamente, usando la tecnica RL
insegnata nel corso (fonte: `10-Tecniche_TORCS.pdf`) più le conoscenze ingegneristiche aggiuntive
necessarie per trasformare le slide in un'implementazione funzionante e sicura.

**In questo documento si mescolano due tipi di fonte, etichettate di conseguenza:**
- **[CORSO]** — contenuto che deriva direttamente dal set di slide del corso. Autorevole ai fini
  dell'esame.
- **[INGEGNERIA]** — conoscenza generale di ingegneria ML aggiunta per rendere implementabile la
  tecnica del corso in questo specifico progetto. Non è materiale d'esame — non attribuirlo al corso.

---

## 1. Vincolo non negoziabile

**La Fase 3 non deve far regredire la Fase 1 o la Fase 2.**

Concretamente:
- **Non** modificare `torcs_env/client.py`, `torcs_env/sensors.py`, `torcs_env/actions.py`, il
  driver `rule_based`, né il modello/pesi BC addestrati. L'RL è **additivo**: un nuovo driver
  (es. `drivers/rl/`) che affianca quelli esistenti.
- Il modello addestrato con BC resta la **baseline di fallback/riferimento**. Il suo attuale
  comportamento di completamento giro e il suo tempo sul giro sono l'asticella che la Fase 3 deve
  eguagliare o superare prima di essere mai promossa a "driver primario".
- Ogni checkpoint RL deve essere valutato con lo `scripts/evaluate.py` esistente, sulle stesse
  metriche già usate per il modello BC (tempo giro, frazione fuori pista, danni/schianti). Nessun
  checkpoint sostituisce il driver attivo a meno che non eguagli BC in sicurezza (nessuno schianto,
  frazione fuori pista comparabile o migliore) **e** non sia peggiore sul tempo giro.
- Se il training RL si destabilizza (diverge, produce una policy peggiore o non sicura),
  l'implementazione deve poter tornare al driver BC senza modifiche di codice altrove.
  (Nota di implementazione: la selezione tramite flag `--driver` citata in origine non esiste
  più — la registry è stata rimossa intenzionalmente; il ritorno al BC è garantito da entry
  point dedicati e paralleli, `scripts/run_agent_rl.py`/`scripts/evaluate_rl.py`, che lasciano
  `run_agent.py`/`evaluate.py` intatti sul driver BC. Deviazione confermata il 2026-07-08.)

---

## 2. Fondamenti di RL **[CORSO]**

Il ciclo agente-ambiente:
1. L'agente osserva lo stato `s_t`.
2. Seleziona un'azione `a_t` seguendo la policy `π(s)`.
3. L'ambiente restituisce il reward `r_{t+1}` e lo stato successivo `s_{t+1}`.
4. L'agente aggiorna la propria policy.

Obiettivo — massimizzare il ritorno scontato atteso:

```
G_t = Σ_{k=0}^{∞} γ^k · r_{t+k+1}
```

- `γ` (fattore di sconto) determina quanto conta il reward futuro rispetto a quello immediato.
  Intervallo tipico: 0.95–0.99.
- Compromesso esplorazione vs. sfruttamento: **ε-greedy** — con probabilità ε si compie un'azione
  casuale (esplorazione), altrimenti si compie la migliore azione nota (sfruttamento).

### Q-Learning (tabulare) **[CORSO]**

```
Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

- `α` = tasso di apprendimento (tipico: 0.1–0.5).
- `max_a' Q(s',a')` codifica l'assunzione di comportamento futuro ottimale — **non** è "l'azione
  che si discosta di più dalle predizioni precedenti" (un punto di confusione ricorrente nelle
  sessioni di studio).
- Non praticabile come algoritmo principale della Fase 3 in questo progetto: lo spazio di stato di
  TORCS è continuo e ad alta dimensionalità (19 rangefinder di pista + velocità + angolo +
  trackPos, ecc.); una Q-table esplode oppure richiede una discretizzazione distruttiva che butta
  via la precisione che il modello BC ha già. Il Q-Learning è utile qui solo come gradino
  concettuale, non come obiettivo implementativo.

### Perché il Deep RL invece **[CORSO]**

- Il Q-Learning tabulare non generalizza tra stati simili e non gestisce nativamente azioni
  continue (`steer` preciso, `accel` fine).
- Una rete neurale approssima `Q` (o direttamente la policy), generalizza tra stati simili e
  gestisce naturalmente spazi di stato/azione continui.
- Compromesso: più potente, ma meno stabile e richiede molti più episodi di training.

---

## 3. Spazio di stato e azione (riepilogo, legato al codice esistente) **[CORSO + INGEGNERIA]**

Corrispondono a ciò che `torcs_env/sensors.py` e `torcs_env/actions.py` già interpretano/serializzano
— non dovrebbe servire nuovo lavoro sui sensori.

**Stato (osservazioni disponibili):**
- `trackPos` (−1 bordo sinistro … +1 bordo destro), `angle` (direzione auto rispetto alla tangente
  di pista)
- `track[19]` — raggi rangefinder (in questo progetto il client li inizializza su −45°…+45°,
  più fitti vicino a 0° — vedi `_DEFAULT_ANGLES` in `torcs_env/client.py`; lo 0°–180° delle
  slide si riferisce alla configurazione SCR di default)
- `speedX`, `speedY`, `speedZ`
- `rpm`, `gear`, `wheelSpinVel[4]`
- opzionali: `focus[5]`, `opponents[36]` (non necessari — gara in solitaria)

**Azione (continua, spazio di default usato da Fase 1/2):**
- `steer ∈ [−1, 1]`, `accel ∈ [0, 1]`, `brake ∈ [0, 1]`
- `gear`: continuare a usare il cambio automatico come già implementato in Fase 1/2 — non
  aggiungere controllo manuale della marcia in Fase 3 a meno che BC/rule-based non lo esponessero
  già, per mantenere lo spazio d'azione identico tra i driver e rendere valido il trasferimento
  di pesi BC→RL.

**[INGEGNERIA]** Dato che la Fase 3 farà il warm-start dall'attore BC (Sezione 6), il layout di
input/output della rete della policy RL **deve corrispondere esattamente** alle feature di input e
alle azioni di output del modello BC. Non re-ingegnerizzare le feature per l'RL; riusare la stessa
funzione di estrazione feature già usata da BC.

---

## 4. Funzione reward

### 4.1 Formula di base **[CORSO]**

```
r_t = v_x · cos(angle) − v_x · |sin(angle)| − v_x · |trackPos|
```

- `+ v_x · cos(angle)` — premia la velocità longitudinale lungo la direzione della pista.
- `− v_x · |sin(angle)|` — penalizza il disallineamento rispetto alla tangente di pista.
- `− v_x · |trackPos|` — penalizza la distanza dal centro pista.

Reward di terminazione:
- Penalità pesante per l'uscita di pista (es. `−100`).
- Bonus opzionale per il completamento del giro.

### 4.2 Questo è un punto di partenza, non un design definitivo **[INGEGNERIA]**

Tony ha confermato che quanto sopra va usato come **baseline da raffinare empiricamente**, non
adottato alla lettera. Raffinamenti concreti che l'implementazione dovrebbe testare e registrare
(una modifica alla volta, così l'effetto di ciascuna è misurabile):

1. **Reward di progresso.** Aggiungere un termine proporzionale alla distanza percorsa lungo la
   linea centrale della pista (non solo la velocità istantanea). Questo punta direttamente alla
   vera metrica di successo del progetto — un giro completo e veloce — ed è la mitigazione
   standard che le slide del corso indicano per il reward hacking (vedi 4.3).
2. **Guardia standing-still / girare sul posto.** Il reward di base può essere raggirato con
   manovre a bassa velocità che evitano il rischio. Penalizzare `speedX` sotto una soglia
   sostenuta per N step, oppure resettare l'episodio se l'auto resta ferma troppo a lungo.
3. **Severità della terminazione fuori pista.** Dato il vincolo rigido del progetto ("nessuno
   schianto, escursioni fuori pista minime"), la penalità di terminazione `−100` andrebbe
   probabilmente tarata *più alta* rispetto ai reward per-step del default da slide, così la
   policy evita con decisione le uscite di pista invece di scambiare una piccola probabilità di
   uscita con una velocità marginalmente più alta.
4. **Coerenza con il warm start BC.** Dato che l'attore parte da pesi BC che già guidano un giro
   completo e sicuro, preferire un reward shaping *potential-based* (non aggiunge bias alla policy
   ottimale) così il fine-tuning RL iniziale non disimpara il comportamento BC sicuro
   nell'inseguimento di guadagni rapidi di reward. In pratica: mantenere le modifiche piccole,
   valutare ogni N step di training rispetto alla baseline BC, e fermare/annullare un run che
   inizia a produrre schianti che il modello BC non aveva.
5. **Normalizzazione.** Mantenere tutte le componenti del reward in intervalli di grandezza
   comparabili, coerenti con la normalizzazione di input già usata per BC ([−1,1] o [0,1] secondo
   il consiglio pratico dello stesso corso, Sezione 6).

Registrare la versione della formula reward usata per ogni run di training (in un
file di configurazione del run) così i risultati sono attribuibili a un design di reward
specifico, non solo a un generico "training RL".

### 4.3 Reward hacking — un rischio specifico di questa formula **[CORSO]**

Le slide avvertono esplicitamente: l'agente può trovare scorciatoie che massimizzano il reward
senza guidare bene — es. girare sul posto per accumulare reward senza rischio di uscire di pista.
Mitigazione indicata dal corso: penalizzare la bassa velocità, aggiungere un termine di reward per
la distanza percorsa. Questo è esattamente il raffinamento #1 e #2 sopra — trattarli come
obbligatori, non opzionali, dato questo specifico modo di fallire già noto per il progetto.

---

## 5. Panoramica degli algoritmi **[CORSO]**

| Algoritmo | Tipo | Azioni continue | Stabilità | Note |
|---|---|---|---|---|
| Q-Learning (tabulare) | value-based, discreto | No (serve discretizzazione) | semplice, comprensibile | buono solo come primo approccio / lane-keeping di base |
| DDPG | actor-critic, off-policy | Nativo | sensibile agli iperparametri, richiede tuning (replay buffer, target net, rumore OU) | ottimo se serve la massima precisione e ci si può permettere il tuning |
| PPO | actor-critic, on-policy | Nativo | molto stabile, raramente diverge | semplice con Stable-Baselines3, ma on-policy → meno sample-efficient, servono più episodi |
| SAC | actor-critic, off-policy | Nativo | off-policy come DDPG ma molto più stabile; auto-regola l'esplorazione via entropia | **raccomandato dalle slide del corso per compiti di controllo continuo come questo** |
| TD3 | actor-critic, off-policy | Nativo | DDPG migliorato (twin critic, aggiornamento ritardato della policy), meno sovrastima | buona alternativa a DDPG |

Componenti comuni del Deep RL (DDPG/SAC/TD3): **replay buffer** (rompe la correlazione temporale
tra transizioni), **reti target** con soft update `τ ≈ 0.001–0.005` (stabilizzano il training
contro un target mobile). PPO usa invece un obiettivo surrogato clippato:

```
L(θ) = min( r_t(θ)·A_t , clip(r_t(θ), 1−ε, 1+ε)·A_t ),  ε ≈ 0.2
```

che limita quanto la policy può cambiare in un singolo update.

---

## 6. Approccio scelto per questo progetto: SAC, con warm-start dall'attore BC

**Algoritmo principale: SAC.** Motivazione — efficienza campionaria off-policy (importante dato
che gli episodi TORCS sono lenti da eseguire, specialmente sul server headless Windows), supporto
nativo per azioni continue, e stabilità di training materialmente migliore di DDPG, il che conta
perché questo progetto non può permettersi che un run divergente corrompa silenziosamente una
baseline funzionante.

### 6.1 Pipeline ibrida BC → RL **[CORSO, questa esatta strategia]**

Le slide raccomandano esplicitamente questo ibrido per il miglior rapporto sforzo/risultato:
1. **Pre-training (già fatto — Fase 2):** la rete addestrata con BC fornisce già una policy
   ragionevole.
2. **Fine-tuning (Fase 3):** inizializzare i pesi dell'attore SAC dalla rete BC, poi continuare il
   training con RL per superare le prestazioni del dimostratore. La convergenza è molto più
   veloce che addestrare SAC da zero perché si parte da una policy già competente.

Note implementative per questo specifico repo:
- Caricare i pesi del modello BC nella rete attore SAC all'inizializzazione. Se l'architettura
  della rete BC differisce da un attore SAC standard (es. diversa attivazione di output),
  aggiungere un layer adattatore invece di riaddestrare l'estrazione delle feature da zero.
- Inizializzare il critic SAC separatamente (non ha un equivalente BC) — qualche migliaio di step
  di warm-up solo-critic prima che inizino gli update congiunti actor-critic può ridurre
  l'instabilità iniziale, dato che l'attore parte "avanti" rispetto a un critic non addestrato.
- Mantenere il file del modello BC intatto, così com'è; salvare il modello raffinato con RL sotto
  un nuovo nome/percorso.

### 6.2 Iperparametri **[INGEGNERIA — non dettagliati per SAC nelle slide del corso]**

Il set di slide del corso fornisce una tabella di iperparametri solo per DDPG (Slide 17). SAC
condivide gran parte degli stessi elementi costitutivi (replay buffer, reti target,
actor-critic), quindi la tabella DDPG è un ancoraggio ragionevole, adattato con i default
standard di SAC:

| Iperparametro | Valore suggerito | Note |
|---|---|---|
| Layer nascosti / unità | 2–3 layer, 64–256 unità | corrisponde alle linee guida generali di Deep RL del corso (Slide 30) |
| Learning rate | 3e-4 (Adam) | default standard di SAC; la tabella DDPG del corso usa 1e-4 |
| Dimensione replay buffer | 100K–1M transizioni | la tabella del corso suggerisce 50K+ per DDPG; SAC beneficia di più transizioni |
| Batch size | 256 | |
| γ (sconto) | 0.99 | l'intervallo del corso è 0.95–0.99 |
| τ (soft update) | 0.005 | la tabella DDPG del corso usa ~0.001; SAC usa comunemente 0.005 |
| Coefficiente di entropia | auto-regolato (target entropy = −dim(spazio azione)) | specifico di SAC, nessun equivalente nel corso |
| Attivazioni | ReLU (nascosti), tanh (output azione) | corrisponde al consiglio pratico del corso (Slide 30) |

Trattarli come una griglia di partenza, non valori fissi — tarare in base alle metriche di
valutazione della Sezione 1.

### 6.3 Integrazione pratica nell'ambiente **[INGEGNERIA]**

- `torcs_env/client.py` è un client UDP SCR grezzo, non un ambiente Gym/Gymnasium. Per
  l'implementazione SAC di Stable-Baselines3, avvolgerlo in un sottile adattatore
  `gymnasium.Env` (`training/rl/torcs_gym_env.py`) che esponga `reset()` → obs e `step(action)` →
  (obs, reward, terminated, truncated, info). **Non modificare il client sottostante** — avvolgerlo.
- Usare `stable_baselines3.SAC` (API identica all'esempio in 3 righe di `PPO` già presente nelle
  slide del corso: `SAC('MlpPolicy', env, ...).learn(total_timesteps=...)`).
- Disabilitare il rendering di TORCS durante il training (headless, come già fatto per Fase 1/2)
  — il corso nota che questo può velocizzare sensibilmente il training.

### 6.4 Tentativo di usare il bot interno "Tita" come sorgente di dimostrazioni **[INGEGNERIA]**

Il bot C++ interno di TORCS "Tita" gira su Corkscrew, con la stessa vettura dello scr_server,
in ~75s — nettamente più veloce sia del BC (121.978s) sia del driver RL residual (127.07s).
È stato tentato (2026-07-14) di estrarne dimostrazioni stato→azione per arricchire il dataset BC,
prima di continuare il lavoro su SAC. Risultato: **abbandonato**, per due motivi indipendenti,
entrambi verificati e non aggirabili senza un intervento fuori scope:

- **Tentativo di ricompilare `tita.dll` con logging aggiunto (accesso ai sorgenti):** i sorgenti
  di Tita esistono in un albero TORCS separato (`gym_torcs/vtorcs-RL-color/src/drivers/tita/`,
  non nell'installazione realmente in uso da `launch_race.py`), con i comandi accessibili a
  `car->_steerCmd`/`_accelCmd`/`_brakeCmd` (`tita.cpp:377-448`). MinGW-w64 è stato installato
  senza problemi (via MSYS2/winget), ma il sistema di build del modulo non è un target isolato:
  dipende da `Make-config` generato da un `./configure` autotools **per Linux/X11**, e la
  dipendenza chiave `plib` (libreria scene-graph richiesta anche da `tita`) non è pacchettizzata
  per Windows/MSYS2 — andrebbe compilata da sorgente, superando ampiamente la soglia di 45 minuti
  fissata prima di tentare. Resta inoltre irrisolta l'incertezza sulla compatibilità ABI tra un
  `tita.dll` ricompilato da quell'albero e l'installazione TORCS binaria realmente in uso (fork
  potenzialmente diverso).
- **Tentativo di usare il replay/telemetria nativa di TORCS come fallback (nessuna ricompilazione
  richiesta):** verificato via codice sorgente che **non esiste alcun meccanismo di replay/log
  utilizzabile in questa build**, indipendentemente dal tempo investito. Il modulo di telemetria
  per-robot (`robottools/rttelem.cpp`, usato potenzialmente da ogni bot incluso Tita) è compilato
  come stub vuoto (ogni funzione avvolta in `#if 0`/`#ifdef later`), con nota esplicita degli
  autori originali *"The telemetry is only working with Linux"* — quindi resterebbe disattivato
  su Windows anche a ricompilazione riuscita. Non esiste inoltre alcun modulo/DLL di replay
  nell'installazione né codice di scrittura `.rpl` in `racemain.cpp`/`raceengine.cpp`.

**Alternative scartate esplicitamente** (non da riconsiderare in futuro senza nuova autorizzazione
esplicita): patch binaria/DLL proxy su `tita.dll` già compilato per intercettare i comandi
(reverse engineering non autorizzato); ricostruzione della traiettoria di Tita dal sensore
"opponents" di un'auto scr_server passiva nella stessa gara (qualità troppo bassa — sensore grezzo
a 36 settori di sola distanza).

**Esito pratico:** il dataset è stato arricchito invece con un ciclo DAgger reale
(`scripts/record_dagger.py`, nuovo script separato) che usa il vecchio `RuleBasedDriver`
(`old_versions_drivers/project_V2/driver.py`, ancora importabile e compatibile con
`torcs_env.sensors.SensorState`/`torcs_env.actions.Action` correnti) come oracolo in ombra durante
il rollout del BC driver reale — non Tita. Vedi il dataset in `data/dagger_bc_*.csv`.

---

## 7. Modalità di fallimento note e mitigazioni **[CORSO]**

| Problema | Causa | Mitigazione |
|---|---|---|
| Reward hacking | l'agente massimizza il reward senza guidare bene (es. girando sul posto) | penalizzare la bassa velocità, premiare la distanza percorsa (Sezione 4.2) |
| Catastrophic forgetting | l'apprendimento azzera la conoscenza precedente | experience replay (SAC lo ha nativamente); variare le condizioni durante il training |
| Minimi locali | l'agente si accontenta di una policy sicura ma sub-ottimale e lenta | bonus di entropia (SAC lo ha nativamente), reset aggressivi in caso di stagnazione |
| Target mobile | il target Q si sposta mentre la rete si allena, causando instabilità | reti target con soft update (SAC lo ha nativamente) |

Dato che SAC affronta già nativamente i problemi di replay/entropia/rete target, il rischio
residuo principale per questo progetto è il **reward hacking contro i vincoli specifici del
progetto** (schianti, uscite di pista) — da qui l'enfasi della Sezione 4.2 su questa particolare
modalità di fallimento.

---

## 8. Consigli pratici ereditati dal corso **[CORSO]**

- Terminazione aggressiva: fuori pista → reset dell'episodio.
- Checkpoint ogni ~50 episodi.
- Monitorare il reward medio **e** la sua deviazione standard per episodio, non solo la media.
- Normalizzare tutti gli input a [−1,1] o [0,1] (già fatto per BC — riusarlo).
- Se il reward non migliora dopo ~100 episodi, cambiare il reward o gli iperparametri prima di
  assumere che più training risolverà il problema.
- Renderizzare/ispezionare periodicamente un episodio per verificare a occhio cosa sta facendo
  davvero la policy.
- Aspettarsi che algoritmi della classe SAC/PPO mostrino risultati decenti in 200K–500K step;
  pianificare di conseguenza rispetto al setup di training a due macchine (Mac M2 / backend MPS
  per il training, PC Windows che esegue il server TORCS headless).

---

## 9. Definizione di "fatto" per la Fase 3

- [ ] `training/rl/torcs_gym_env.py` — wrapper Gymnasium attorno al client SCR esistente, nessuna
      modifica a `torcs_env/*`.
- [ ] `training/rl/train_sac.py` — script di training, attore inizializzato dai pesi BC della
      Fase 2, versione del reward configurabile, checkpoint ogni ~50 episodi.
- [ ] `drivers/rl/driver.py` — nuova classe driver che implementa la stessa interfaccia dei driver
      `rule_based`/`bc_model`, caricabile via `run_agent.py --driver rl_model`.
- [ ] Funzione reward versionata e registrata per ogni run di training.
- [ ] `scripts/evaluate.py` eseguito sul driver RL, confrontato fianco a fianco con le metriche del
      driver BC esistente (tempo giro, frazione fuori pista, numero di schianti) — l'RL viene
      promosso a default solo se eguaglia o supera BC in sicurezza e non è peggiore sul tempo giro.
- [ ] Documentazione aggiornata con: stato Fase 3, algoritmo usato, versione del reward, miglior tempo
      giro, e conferma esplicita che i driver Fase 1/2 restano non modificati e funzionanti.

---

## Riferimenti

- Set di slide del corso: `10-Tecniche_TORCS.pdf` (32 slide — ambiente/sensori, funzione reward,
  Q-Learning, Deep RL/DDPG/PPO, Imitation Learning, insidie comuni).
- `stable-baselines3.readthedocs.io` — riferimento API SAC/PPO/DDPG/TD3.
- Risorse esterne citate dal corso: `github.com/YurongYou/rlTORCS` (esempi di Deep RL su TORCS),
  `amslaurea.unibo.it` (Galletti, 2019 — tesi di confronto DDPG vs PPO).
