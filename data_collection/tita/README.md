# Raccolta dati BC dal bot tita

Logging hardcoded, attivo ad ogni tick a prescindere da come la gara viene
avviata (GUI quickrace/practice incluso). Nessun file esistente di
`torcs_env/`, `drivers/`, `training/` o degli script BC e' stato toccato.

## Architettura: DLL proxy

Il primo tentativo (istrumentare direttamente `tita.cpp` e ricompilare tutta
la logica di guida) si e' scontrato con un problema irrisolto: l'albero
sorgente in `old_versions_drivers/project_V1/torcs/gym_torcs/vtorcs-RL-color`
non corrisponde esattamente al motore TORCS realmente installato in
`U:\AI-Partition\torcs\torcs\` (probabilmente un fork/patch-level diverso —
confermato con evidenze concrete: simboli C++ vs C, e uno struct `tPrivCar`
che non combaciava). Con un fix mirato la telemetria letta e' risultata
corretta (verificata contro `car1-ow1.xml`: massa, carburante, rpm motore
tutti plausibili), ma il cambio marce restava bloccato in folle per una
causa nel motore fisico non identificabile senza il suo sorgente (e' in
`modules/simu/*.dll`, non incluso in questo repo).

**Soluzione adottata**: invece di far guidare l'auto al codice ricompilato,
un piccolo **DLL proxy** (`tita_proxy.cpp`) carica il `tita.dll` originale
(rinominato `tita_real.dll`, binario precompilato mai modificato) e inoltra
ad esso *tutte* le callback dell'interfaccia robot TORCS — quindi la guida è
al 100% quella reale/originale di tita (stessi tempi sul giro verificati:
identico all'originale). Il proxy si aggancia solo in coda a `rbNewRace` /
`rbDrive` / `rbShutdown` per scrivere il CSV con il logger BC (che legge i
suoi sensori in modo indipendente, gia' validato).

## File aggiunti (sorgente TORCS)

In `old_versions_drivers/project_V1/torcs/gym_torcs/vtorcs-RL-color/src/drivers/tita/`:

- `bc_sensors.h` / `bc_sensors.cpp` — port 1:1 dei 19 raggi range-finder di
  `src/drivers/scr_server/sensors.cpp` (classe rinominata `BcSensors`).
- `bc_logger.h` / `bc_logger.cpp` — apre un CSV per sessione e scrive una riga
  per tick con lo schema del dataset BC esistente:
  `angle,speed,speedY,speedZ,trackPos,track_0..track_18,rpm,gear,steer,accel,brake,gear_cmd`
  (stesso ordine/nomi di `data/driver_*.csv` e di `INPUT_COLS` in
  `scripts/train/train_bc_dagger.py`). Nota: la colonna `gear_cmd` non e' allineata
  correttamente (valori spazzatura) per una struct non ancora del tutto
  risolta — **non e' usata dal training** (`train_bc_dagger.py` usa solo
  `steer,accel,brake` come target), quindi non inquina il dataset. Tutte le
  altre colonne sono verificate corrette su un giro completo.
- `tita_proxy.cpp` — il DLL proxy descritto sopra (entry point `tita()`,
  inoltra `rbNewTrack/rbNewRace/rbEndRace/rbDrive/rbPitCmd/rbShutdown` al
  `tita_real.dll` originale, con hook di logging su NewRace/Drive/Shutdown).
- `tita_mingw_abi_fix.h` — dichiarazioni `extern "C"` per le funzioni
  `GfParm*`/`_tgf_win_strdup` (necessarie solo per una build MinGW, guardate
  `#ifdef __GNUC__`, non toccano la build MSVC/VC6 originale).

`tita.cpp`/`mycar.cpp`/`trackdesc.cpp` (la logica di guida originale) **non
sono piu' usati per il DLL in produzione** — restano nel repo con le
istruzioni BC gia' inserite (vedi git-diff-like modifiche precedenti) nel
caso servano in futuro per un debug piu' approfondito del motore fisico, ma
il DLL effettivamente installato oggi e' il proxy.

`src/interfaces/car.h`: `tPrivCar` ha 2 campi puntatore di padding aggiunti
prima di `carHandle`, guardati `#ifdef __GNUC__` — necessari perche' il
logger BC legge `car->_gear`/`car->_enginerpm`/etc (in `tPrivCar`) per la
telemetria, anche se il proxy non guida piu' l'auto.

## Bug corretto: offset di tCarCtrl (steer/accel/brake/gear_cmd)

**Scoperto il 2026-07-15**: le colonne `steer,accel,brake,gear_cmd` lette da
`car->ctrl` (tCarCtrl) erano lette da un offset di memoria sbagliato — lo
stesso tipo di problema gia' risolto per `tPrivCar` (vedi sopra), ma mai
applicato a `ctrl`. Sintomo noto ma sottovalutato: `gear_cmd` con valori
spazzatura (es. `1057300156`), liquidato all'epoca come "non usato, innocuo".
In realta' l'intera struct `ctrl` era disallineata, quindi **anche
`steer`/`accel`/`brake` — usati come target di training — erano corrotti**,
non solo `gear_cmd`.

Prova diretta: durante il lancio (0->80 km/h in ~1.8s) i valori registrati
mostravano `accel≈0.01` e `brake≈0.5-1.0` — fisicamente impossibile (quella
accelerazione richiede piena erogazione, non freno a tavoletta).

**Diagnosi**: scansione empirica della memoria (stesso metodo gia' usato per
`tPrivCar`) attorno al puntatore `&car->ctrl` calcolato dal nostro header,
correlando i quadrupli candidati (steer,accelCmd,brakeCmd,clutchCmd,gear)
contro il comportamento reale dell'auto in una fase di lancio + frenata nota.
Risultato inequivocabile: il vero `ctrl` inizia **4 byte dopo** rispetto a
dove lo calcolava il nostro header (293/301 tick di accelerazione
combaciavano esattamente all'offset +4, contro 0 su tutti gli altri 100
offset testati).

**Fix**: aggiunti 4 byte di padding (`_mingw_abi_ctrl_pad`) in `tCarElt` tra
`priv` e `ctrl`, guardati `#ifdef __GNUC__` (stesso pattern del fix
`tPrivCar`), in `src/interfaces/car.h`. Non serve individuare esattamente
dove nella catena `info/pub/race/priv` mancano quei 4 byte: il padding
inserito appena prima di `ctrl` corregge l'offset a valle senza toccare
nient'altro.

**Verificato dopo il fix** (giro headless completo): lancio con
`accel=1.0,brake=0.0`; frenata forte con `accel→0,brake→0.63`; `gear_cmd` ora
combacia sempre con la marcia reale (niente piu' interi spazzatura).

**Impatto sui dati gia' raccolti**: le sessioni registrate PRIMA di questo
fix (incluse le 3 sessioni dell'11 giugno 2026-07-15, ~11 giri totali) hanno
`steer/accel/brake` corrotti e vanno considerate **inutilizzabili per il
training** — le colonne sensoriali (angle/speed/trackPos/track_*/rpm/gear)
restano valide (lette da `priv`, mai affette da questo bug), ma le azioni no.
Vanno ri-registrate con la build del proxy successiva a questo fix.

## Dove finiscono i CSV

Directory `bc_logs/` **relativa alla working directory di `wtorcs.exe`**
(quindi normalmente `U:\AI-Partition\torcs\torcs\bc_logs\`), creata
automaticamente al primo giro. Override possibile con la variabile
d'ambiente `TITA_BC_LOG_DIR` prima di lanciare `wtorcs.exe`.

Nome file: `tita_bc_<indice_auto>_<timestamp>.csv`, un file nuovo per ogni
`newRace` (ogni volta che parte una gara), quindi nessuna sovrascrittura tra
sessioni.

## Build (Windows, toolchain effettivamente usato: MinGW-w64 32-bit via MSYS2)

Il progetto VC6/`tita.dsp` incluso nel repo non e' stato verificato contro
questo motore (vedi sopra); il DLL attualmente in uso e' stato compilato con
MinGW-w64 32-bit (pacchetto `mingw-w64-i686-gcc`, installato via
`pacman` di MSYS2 — leggero, ~490MB, reversibile con `pacman -R`).

Per ricompilare (solo se serve modificare il proxy o il logger):

```bash
export PATH="/c/msys64/mingw32/bin:$PATH"
cd old_versions_drivers/project_V1/torcs/gym_torcs/vtorcs-RL-color

INC="-I. -Isrc/interfaces -Isrc/libs/tgf -Isrc/libs/robottools \
     -Isrc/libs/portability -Isrc/libs -Isrc/drivers/tita \
     -Isrc/windows -Isrc/windows/include \
     -include src/drivers/tita/tita_mingw_abi_fix.h"

for f in tita_proxy bc_sensors bc_logger; do
  g++ -m32 -O2 -c src/drivers/tita/$f.cpp -o /tmp/$f.o $INC
done

# import lib per robottools.dll (generata una tantum da objdump+dlltool
# sul robottools.dll reale in U:\AI-Partition\torcs\torcs\; non serve tgf.dll
# perche' il proxy/logger non chiamano funzioni GfParm*)

g++ -m32 -shared -o tita.dll \
  /tmp/tita_proxy.o /tmp/bc_sensors.o /tmp/bc_logger.o \
  src/drivers/tita/tita.def \
  -Wl,--defsym,__Z19RtTrackSideTgAngleLP10tTrkLocPos=_RtTrackSideTgAngleL \
  -L<cartella con librobottools.a> -lrobottools \
  -static -static-libgcc -static-libstdc++
```

Nota sul `.def`: MinGW non digerisce la sintassi VC6 (`LIBRARY` senza nome +
`SECTIONS`) — serve un `.def` minimale con solo `EXPORTS\n\ttita`.

Nota sul `--defsym`: gli header di questo fork dichiarano `RtTrackSideTgAngleL`
senza `extern "C"`, quindi GCC genera un simbolo C++-mangled che non esiste
nel `robottools.dll` reale (che esporta il nome plain). Il `--defsym` fa
puntare il simbolo mangled a quello vero.

Deploy:
1. `tita_real.dll` = copia del `tita.dll` originale/funzionante (mai
   ricompilato, va preso da un backup).
2. `tita.dll` (appena compilato sopra, il proxy) va in
   `U:\AI-Partition\torcs\torcs\drivers\tita\tita.dll`.
3. Entrambi i file stanno nella stessa cartella `drivers/tita/`; TORCS
   carica solo `tita.dll` per nome, `tita_real.dll` viene caricato a runtime
   dal proxy stesso via `LoadLibraryA("drivers/tita/tita_real.dll")`
   (percorso relativo alla working directory di `wtorcs.exe`).

## Conversione al formato dataset BC

```
python data_collection/tita/convert_tita_csv.py --input "bc_logs/tita_bc_*.csv"
```

Valida le colonne, applica la stessa pulizia usata da
`scripts/train/train_bc_dagger.py` (`|trackPos| < 0.95`, `|speed| > 1.0`) e scrive
un file per sessione in `data_collection/tita/converted/`, senza toccare
`data/` ne' i CSV originali. L'unione al dataset esistente resta manuale, a
tua discrezione (copia in `data/` oppure passa
`--original "data_collection/tita/converted/*.csv"` a
`scripts/train/train_bc_dagger.py`).
