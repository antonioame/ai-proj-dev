# Raccolta dati BC dal bot tita

Logging hardcoded, attivo ad ogni tick a prescindere da come la gara viene
avviata (GUI quickrace/practice incluso). Nessun file esistente di
`torcs_env/`, `drivers/`, `training/` o degli script BC è stato toccato.

## Architettura: DLL proxy

Istrumentare direttamente `tita.cpp` e ricompilare la logica di guida non ha
funzionato: l'albero sorgente in
`old_versions_drivers/project_V1/torcs/gym_torcs/vtorcs-RL-color` non
corrisponde al motore TORCS installato in `U:\AI-Partition\torcs\torcs\`
(fork/patch-level diverso: simboli C++ vs C, struct `tPrivCar` non
combaciante). La telemetria letta risultava corretta dopo il fix, ma il
cambio marce restava bloccato in folle per una causa nel motore fisico
(`modules/simu/*.dll`, non incluso nel repo, sorgente non disponibile).

**Soluzione**: un DLL proxy (`tita_proxy.cpp`) carica il `tita.dll`
originale (rinominato `tita_real.dll`, binario precompilato mai modificato)
e inoltra ad esso tutte le callback dell'interfaccia robot TORCS: la guida
resta al 100% quella originale (stessi tempi sul giro verificati). Il proxy
si aggancia in coda a `rbNewRace`/`rbDrive`/`rbShutdown` solo per scrivere il
CSV col logger BC.

## File aggiunti (sorgente TORCS)

In `old_versions_drivers/project_V1/torcs/gym_torcs/vtorcs-RL-color/src/drivers/tita/`:

- `bc_sensors.h/.cpp`: port 1:1 dei 19 raggi range-finder di
  `src/drivers/scr_server/sensors.cpp` (classe rinominata `BcSensors`).
- `bc_logger.h/.cpp`: un CSV per sessione, una riga per tick, schema
  `angle,speed,speedY,speedZ,trackPos,track_0..track_18,rpm,gear,steer,accel,brake,gear_cmd`
  (stesso ordine di `data/driver_*.csv` e `INPUT_COLS` in
  `scripts/train/train_bc_dagger.py`). `gear_cmd` non è allineata
  correttamente (valori spazzatura) ma non è usata dal training: le altre
  colonne sono verificate corrette su un giro completo.
- `tita_proxy.cpp`: il DLL proxy (entry point `tita()`, inoltra
  `rbNewTrack/rbNewRace/rbEndRace/rbDrive/rbPitCmd/rbShutdown` al
  `tita_real.dll`, con hook di logging su NewRace/Drive/Shutdown).
- `tita_mingw_abi_fix.h`: dichiarazioni `extern "C"` per `GfParm*`/
  `_tgf_win_strdup`, necessarie solo per la build MinGW (`#ifdef __GNUC__`,
  non tocca la build MSVC/VC6 originale).

`tita.cpp`/`mycar.cpp`/`trackdesc.cpp` (logica di guida originale) non sono
più usati per il DLL in produzione. Restano nel repo per un eventuale debug
futuro del motore fisico.

`src/interfaces/car.h`: `tPrivCar` ha 2 campi puntatore di padding aggiunti
prima di `carHandle` (`#ifdef __GNUC__`), necessari perché il logger BC legge
`car->_gear`/`car->_enginerpm`/etc per la telemetria, anche se il proxy non
guida più l'auto.

## Bug corretto: offset di tCarCtrl (steer/accel/brake/gear_cmd)

Le colonne `steer,accel,brake,gear_cmd` lette da `car->ctrl` (tCarCtrl)
venivano lette da un offset di memoria sbagliato, lo stesso problema già
risolto per `tPrivCar`, ma non applicato a `ctrl`. Il sintomo (`gear_cmd` con
valori spazzatura) era stato liquidato come "non usato, innocuo", ma l'intera
struct era disallineata: anche `steer/accel/brake` (i target di training)
erano corrotti.

Prova diretta: durante il lancio (0->80 km/h in circa 1.8s) i valori
registrati mostravano `accel` circa 0.01 e `brake` circa 0.5-1.0, fisicamente impossibile.

**Diagnosi**: scansione della memoria attorno a `&car->ctrl`, correlando i
quadrupli candidati (steer,accelCmd,brakeCmd,clutchCmd,gear) col
comportamento reale dell'auto in una fase di lancio+frenata nota. Il vero
`ctrl` inizia 4 byte dopo rispetto a dove lo calcolava il nostro header
(293/301 tick combaciavano esattamente all'offset +4, contro 0 su tutti gli
altri 100 offset testati).

**Fix**: 4 byte di padding (`_mingw_abi_ctrl_pad`) in `tCarElt` tra `priv` e
`ctrl` (`#ifdef __GNUC__`, in `src/interfaces/car.h`): corregge l'offset a
valle senza dover individuare dove esattamente nella catena
`info/pub/race/priv` mancano quei 4 byte.

**Verificato dopo il fix**: lancio con `accel=1.0,brake=0.0`, frenata forte
con `accel→0,brake→0.63`, `gear_cmd` ora combacia sempre con la marcia reale.

**Impatto sui dati già raccolti**: le sessioni registrate prima di questo fix
(incluse le 3 sessioni dell'11 giugno, circa 11 giri totali) hanno
`steer/accel/brake` corrotti e sono **inutilizzabili per il training**. Le
colonne sensoriali (angle/speed/trackPos/track_*/rpm/gear) restano valide
(lette da `priv`, non affette dal bug). Vanno ri-registrate con la build del
proxy successiva a questo fix.

## Dove finiscono i CSV

Directory `bc_logs/` relativa alla working directory di `wtorcs.exe`
(normalmente `U:\AI-Partition\torcs\torcs\bc_logs\`), creata automaticamente
al primo giro. Override con la variabile d'ambiente `TITA_BC_LOG_DIR` prima
di lanciare `wtorcs.exe`.

Nome file: `tita_bc_<indice_auto>_<timestamp>.csv`, un file nuovo per ogni
`newRace`, quindi nessuna sovrascrittura tra sessioni.

## Deploy

Il DLL in uso è compilato con MinGW-w64 32-bit via MSYS2 (il progetto
VC6/`tita.dsp` incluso nel repo non è stato verificato contro questo motore).

1. `tita_real.dll` = copia del `tita.dll` originale/funzionante (mai
   ricompilato, va preso da un backup).
2. `tita.dll` (il proxy compilato) va in
   `U:\AI-Partition\torcs\torcs\drivers\tita\tita.dll`.
3. Entrambi i file nella stessa cartella `drivers/tita/`: TORCS carica solo
   `tita.dll` per nome, `tita_real.dll` viene caricato a runtime dal proxy
   via `LoadLibraryA("drivers/tita/tita_real.dll")` (percorso relativo alla
   working directory di `wtorcs.exe`).

## Conversione al formato dataset BC

```
python data_collection/tita/convert_tita_csv.py --input "bc_logs/tita_bc_*.csv"
```

Valida le colonne, applica la stessa pulizia di
`scripts/train/train_bc_dagger.py` (`|trackPos| < 0.95`, `|speed| > 1.0`) e
scrive un file per sessione in `data_collection/tita/converted/`, senza
toccare `data/` né i CSV originali. L'unione al dataset esistente resta
manuale (copia in `data/`, oppure `--original
"data_collection/tita/converted/*.csv"` a `scripts/train/train_bc_dagger.py`).
