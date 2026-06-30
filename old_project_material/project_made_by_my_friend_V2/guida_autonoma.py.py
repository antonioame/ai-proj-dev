"""
Progetto diIntelligenza Artificiale 
-----------------------------------------------------------------------------
Descrizione: Modulo di Guida Autonoma per il simulatore TORCS (The Open Racing Car Simulator).
Algoritmo: Rete Neurale MLP (Multi-Layer Perceptron) addestrata ad imitazione (Imitation Learning).
Valore Loss di Validazione: 0.0172

Caratteristiche del dataset di addestramento:
- Target dello sterzo normalizzato (moltiplicato per un fattore di scala pari a 0.6)
- Filtraggio campioni per addestramento su traiettorie pulite (|trackPos| < 0.9)
- Comportamento su pista: guida consistente e riproducibile su più giri (~100s per giro)
"""

import time
import numpy as np
import joblib
import torch
import torch.nn as nn
import snakeoil3_jm2 as snakeoil3


class DrivingNet(nn.Module):
    """
    Rete neurale MLP (Multi-Layer Perceptron) per la predizione dei comandi di guida.
    La rete elabora i sensori del tracciato e predice: angolo di sterzo, accelerazione/freno e marcia.
    """
    def __init__(self, dim_ingresso: int, numero_marce: int = 8):
        super().__init__()
        # Backbone comune per l'estrazione delle feature dallo stato del tracciato
        self.backbone = nn.Sequential(
            nn.Linear(dim_ingresso, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128),    nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),     nn.ReLU(),
        )
        # Teste di output dedicate ai singoli controlli (Multi-Task Learning)
        self.head_steer       = nn.Linear(64, 1)
        self.head_accel_brake = nn.Linear(64, 2)
        self.head_gear        = nn.Linear(64, numero_marce)

    def forward(self, dati_ingresso):
        strato_nascosto = self.backbone(dati_ingresso)
        return (
            torch.tanh(self.head_steer(strato_nascosto)),
            torch.sigmoid(self.head_accel_brake(strato_nascosto)),
            self.head_gear(strato_nascosto),
        )


def build_input_vector(sensori_stato, colonne_selezionate):
    """
    Costruisce il vettore dei dati di input normalizzati per la rete neurale.
    Estrae le letture dei sensori di telemetria da TORCS e le mappa in ordine coerente.
    """
    # Estraiamo i 19 sensori di distanza del tracciato (raggio visivo)
    distanze_tracciato = sensori_stato.get("track", [200.0] * 19)
    # Velocità di rotazione delle 4 ruote
    rotazione_ruote = sensori_stato.get("wheelSpinVel", [0.0] * 4)
    
    # Gestione di eventuali letture corrotte o mancanti
    if len(distanze_tracciato) != 19:  distanze_tracciato  = [200.0] * 19
    if len(rotazione_ruote) != 4:  rotazione_ruote = [0.0] * 4
    
    dizionario_caratteristiche = {}
    # Mappatura dei sensori del tracciato
    for i in range(19):
        dizionario_caratteristiche[f"track_{i}"] = float(distanze_tracciato[i])
    # Mappatura della velocità delle ruote
    for i in range(4):
        dizionario_caratteristiche[f"wheelSpin_{i}"] = float(rotazione_ruote[i])
        
    # Aggiunta di altre variabili fisiche e dinamiche del veicolo
    dizionario_caratteristiche["speedX"]   = float(sensori_stato.get("speedX", 0))
    dizionario_caratteristiche["speedY"]   = float(sensori_stato.get("speedY", 0))
    dizionario_caratteristiche["speedZ"]   = float(sensori_stato.get("speedZ", 0))
    dizionario_caratteristiche["trackPos"] = float(sensori_stato.get("trackPos", 0))
    dizionario_caratteristiche["angle"]    = float(sensori_stato.get("angle", 0))
    dizionario_caratteristiche["rpm"]      = float(sensori_stato.get("rpm", 0))
    
    # Ritorniamo il vettore NumPy ordinato secondo le colonne usate in fase di addestramento (Feature Selection)
    return np.array([dizionario_caratteristiche[colonna] for colonna in colonne_selezionate], dtype=np.float32)


def auto_gear(giri_motore, marcia_corrente):
    """
    Funzione di controllo euristico di riserva per la selezione automatica della marcia.
    Utilizza soglie fisse di giri al minuto (RPM) del motore per cambiare rapporto.
    """
    if marcia_corrente < 1: return 1
    # Soglia superiore per passare al rapporto successivo (Upshift)
    if giri_motore > 8000 and marcia_corrente < 6: return marcia_corrente + 1
    # Soglia inferiore per scalare il rapporto (Downshift)
    if giri_motore < 3500 and marcia_corrente > 1: return marcia_corrente - 1
    return marcia_corrente


def main():
    print("[INFO] Caricamento del modello di guida e dello scaler...")
    # Carichiamo l'oggetto scaler precedentemente salvato in fase di preprocessing dei dati
    dati_scaler = joblib.load("driving_scaler.pkl")
    media_normalizzazione = dati_scaler["mean"]
    deviazione_standard = dati_scaler["std"]
    colonne_input = dati_scaler["input_cols"]
    offset_marce = dati_scaler["gear_offset"]

    # Selezione del dispositivo hardware per l'inferenza (GPU se disponibile, altrimenti CPU)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializziamo l'architettura della rete e carichiamo i pesi addestrati (.pt)
    modello_guida = DrivingNet(dim_ingresso=len(colonne_input)).to(dispositivo)
    modello_guida.load_state_dict(torch.load("driving_model.pt", map_location=dispositivo))
    modello_guida.eval()  # Impostiamo la rete in modalità inferenza (disabilita Dropout)
    print(f"[INFO] Modello caricato con successo ed eseguito su: {dispositivo}.")

    # Inizializzazione della connessione client socket con il server di simulazione TORCS
    client_torcs = snakeoil3.Client(p=3001, vision=False)
    client_torcs.get_servers_input()
    print("[INFO] Connessione a TORCS stabilita. Avvio del loop di controllo...\n")

    # Inizializzazione variabili per la gestione dello stato del veicolo
    tempo_ultimo_cambio = 0.0
    marcia_attuale = 1
    sterzata_precedente = 0.0
    contatore_passi = 0

    # Liste di storicizzazione per rilevare eventuali anomalie/congelamenti del simulatore (Watchdog)
    storico_posizioni_tracciato = []
    storico_velocita = []

    # Parametri e contatori per l'algoritmo di recupero in caso di fuoripista (Recovery Controller)
    contatore_fuoripista = 0
    modalita_recupero = None
    passo_inizio_recupero = 0
    direzione_recupero_precedente = 0
    contatore_post_recupero = 0

    # Variabili per il monitoraggio delle performance sul giro (Lap Timer)
    tempo_giro_precedente = 0.0
    giro_corrente = 0
    passo_inizio_giro = 0

    # Velocità massima di sicurezza impostata per scenari particolari (Speed Cap)
    limite_velocita = 80.0

    while True:
        try:
            # Acquisizione pacchetto dati aggiornato dal server di TORCS
            client_torcs.get_servers_input()
            sensori = client_torcs.S.d

            # Calcolo del tempo sul giro corrente e rilevazione completamento del tracciato
            tempo_giro_corrente = sensori.get("curLapTime", 0.0)
            if tempo_giro_corrente < tempo_giro_precedente - 1.0:
                # Se il tempo corrente si azzera improvvisamente, abbiamo completato un giro!
                giro_corrente += 1
                passi_trascorsi = contatore_passi - passo_inizio_giro
                print(f"\n  === [STATISTICHE] GIRO {giro_corrente} CONCLUSO "
                      f"(Tempo: {tempo_giro_precedente:.0f}s, Passi/Frequenza: {passi_trascorsi} step) ===\n")
                passo_inizio_giro = contatore_passi
            tempo_giro_precedente = tempo_giro_corrente

            # --- FASE DI INFERENZA DELLA RETE NEURALE ---
            # 1. Costruzione del vettore di input dai sensori grezzi
            vettore_input = build_input_vector(sensori, colonne_input)
            # 2. Normalizzazione z-score (standardizzazione) basata su media e deviazione standard del dataset
            input_normalizzato = (vettore_input - media_normalizzazione) / deviazione_standard
            # 3. Conversione in PyTorch Tensor e aggiunta della dimensione batch (unsqueeze)
            tensor_input = torch.from_numpy(input_normalizzato).unsqueeze(0).to(dispositivo)

            # Disabilitiamo il calcolo dei gradienti per velocizzare l'esecuzione (no_grad)
            with torch.no_grad():
                predizione_sterzo, predizione_pedali, logits_marcia = modello_guida(tensor_input)

            # 4. Post-processing degli output della rete neurale
            # Lo sterzo era scalato a 0.6 nel dataset originale; applichiamo un guadagno correttivo di 1.8 per aumentare la reattività dinamica
            angolo_sterzo = float(predizione_sterzo.item()) * 1.8
            acceleratore = float(predizione_pedali[0, 0].item())
            freno = float(predizione_pedali[0, 1].item())
            
            # Classificazione per la marcia: prendiamo l'indice con valore massimo (argmax) e togliamo l'offset di codifica
            marcia_predetta = int(logits_marcia.argmax(dim=1).item()) - offset_marce

            # --- LETTURA SENSORI CRITICI DI TELEMETRIA ---
            velocita_attuale = sensori.get("speedX", 0)  # Velocità longitudinale (asse X)
            sensori_distanza = sensori.get("track", [200]*19)  # 19 sensori di distanza a ventaglio
            distanza_fronte = sensori_distanza[9]  # Sensore centrale (indice 9, guarda dritto in avanti)
            posizione_tracciato = sensori.get("trackPos", 0)  # Offset rispetto alla mezzeria (-1 a sinistra, +1 a destra)


            # --- SISTEMA WATCHDOG (ANTIFREEZE) ---
            # Evita loop bloccati se il simulatore si arresta o crasha (attivo dopo i primi 100 step)
            if contatore_passi > 100:
                storico_posizioni_tracciato.append(posizione_tracciato)
                storico_velocita.append(velocita_attuale)
                
                # Manteniamo una finestra temporale di analisi di 50 campioni (FIFO)
                if len(storico_posizioni_tracciato) > 50:
                    storico_posizioni_tracciato.pop(0)
                    storico_velocita.pop(0)
                    
                    # Se posizione e velocità rimangono perfettamente identiche per 50 passi, la simulazione è congelata
                    if len(set(storico_posizioni_tracciato)) == 1 and len(set(storico_velocita)) == 1:
                        print("\n[WARNING] Rilevato congelamento della simulazione TORCS. Disconnessione socket in corso...")
                        break

            # FASE 1: Transitorio di partenza / Start-up
            # Nei primi 80 passi temporali applichiamo un controllo euristico per superare l'inerzia iniziale
            if contatore_passi < 80:
                acceleratore = 1.0
                freno = 0.0
                angolo_sterzo = angolo_sterzo * 0.5  # Attenuazione dello sterzo per evitare sbandate immediate
                if velocita_attuale < 5:    marcia_attuale = 1
                elif velocita_attuale < 15: marcia_attuale = 2
                else:                       marcia_attuale = 3
            # FASE 2 & FASE 3: Controllore di Velocità Longitudinale (Cruise Control Reattivo)
            else:
                # 1. Calcolo del raggio visivo efficace (Lookahead) analizzando l'arco visivo frontale (sensori da 5 a 13)
                sguardo_avanti = max(
                    sensori_distanza[5], sensori_distanza[6], sensori_distanza[7], 
                    sensori_distanza[8], sensori_distanza[9], sensori_distanza[10], 
                    sensori_distanza[11], sensori_distanza[12], sensori_distanza[13]
                )

                # Velocità target calibrate sul log: i tornanti DEVONO restare bassi
                # perché la rete neurale sterza male ad alta velocità in quelle curve.
                # Alzare i tornanti da 63 a 78 ha causato 4 uscite invece di 1.
                # Guadagno applicato solo nelle curve medie/larghe dove è provato sicuro.
                if sguardo_avanti > 170.0:
                    velocita_target = 199.0  # Rettilineo principale
                elif sguardo_avanti > 120.0:
                    velocita_target = 195.0  # Rettilineo secondario (+3)
                elif sguardo_avanti > 80.0:
                    velocita_target = 176.0  # Curva larga (+4, era 172)
                elif sguardo_avanti > 50.0:
                    velocita_target = 143.0  # Curva media (+5, era 138)
                elif sguardo_avanti > 30.0:
                    velocita_target = 112.0  # Curva stretta (+2, era 110)
                else:
                    velocita_target = 65.0   # Tornante: solo +2 da 63, sicuro confermato dal log

                # 2. Correzione di sicurezza sulla velocità target se il veicolo devia eccessivamente dalla mezzeria
                if abs(posizione_tracciato) > 0.55:
                    # Inizia a limitare già a 0.55 per smorzare l'oscillazione laterale prima che degeneri
                    velocita_target = min(velocita_target, 115.0)
                if abs(posizione_tracciato) > 0.65:
                    velocita_target = min(velocita_target, 85.0)
                if abs(posizione_tracciato) > 0.78:
                    velocita_target = min(velocita_target, 55.0)

                # 3. Calcolo teorico dello spazio di arresto richiesto basato sull'equazione cinematica s = v^2 / (2 * a) + offset
                # Dove la decelerazione stimata a = 116 m/s^2 e l'offset di sicurezza è 5.0 metri
                spazio_frenata_richiesto = (velocita_attuale ** 2) / 232.0 + 5.0

                # 4. Modulazione dell'accelerazione e della frenata (Controllore Proporzionale con ABS e EBD simulati)
                if velocita_attuale < velocita_target:
                    differenza_velocita = velocita_target - velocita_attuale
                    
                    # In rettilineo o curve ampie (>60m) applichiamo massima accelerazione
                    if sguardo_avanti > 60.0:
                        acceleratore = 1.0
                    elif sguardo_avanti <= 30.0:
                        # Tornante: acc più decisa per non perdere tempo a 64 km/h con acc=0.09
                        # Il log mostra diff=1 km/h -> acc=0.2, troppo bassa. Usiamo divisore 3.
                        acceleratore = min(0.55, differenza_velocita / 3.0)
                    else:
                        # Curve medie: P control
                        acceleratore = min(1.0, differenza_velocita / 5.0)
                    freno = 0.0
                else:
                    differenza_velocita = velocita_attuale - velocita_target
                    acceleratore = 0.0
                    
                    # Frenata anticipata proattiva ad alte velocità
                    if velocita_attuale > 160.0:
                        sguardo_frenata = min(sguardo_avanti, distanza_fronte * 1.05)
                    else:
                        sguardo_frenata = sguardo_avanti
                    
                    if sguardo_frenata < spazio_frenata_richiesto:
                        # ABS Dinamico
                        if velocita_attuale > 140.0:
                            frenata_massima = 0.62
                        elif velocita_attuale > 90.0:
                            frenata_massima = 0.73
                        else:
                            frenata_massima = 0.84
                        
                        # EBD: riduce frenata se in sterzata
                        if abs(angolo_sterzo) > 0.08:
                            frenata_consentita = frenata_massima - (abs(angolo_sterzo) - 0.08) * 0.75
                            frenata_massima = max(0.38, min(frenata_massima, frenata_consentita))
                        
                        # Frenata progressiva: denominatore alzato da 25 a 38 per evitare
                        # le frenate doppie viste a step 1550 (176->118 km/h in 50 step)
                        margine_spazio = spazio_frenata_richiesto - sguardo_frenata
                        intensita_frenata = min(1.0, margine_spazio / 38.0)
                        freno = frenata_massima * intensita_frenata
                    else:
                        # Coasting con throttle: soglia allargata a 0.55 (era 0.45)
                        # e throttle aumentato a 0.35 (era 0.20).
                        # Risolve i coasting inutili a 142-148 km/h (step 400, 3650, 3700, 4050)
                        if abs(posizione_tracciato) < 0.55 and differenza_velocita < 14.0:
                            acceleratore = min(0.35, differenza_velocita / 8.0)
                        freno = 0.0

                # 5. Controllo di Trazione (TCS - Traction Control System)
                # Ripristinati i valori originali: il TCS meno aggressivo della v2
                # ha permesso troppa accelerazione in uscita dai tornanti causando 4 uscite.
                if abs(angolo_sterzo) > 0.10:
                    if marcia_attuale < 3:
                        fattore_tcs = 1.45
                    elif marcia_attuale == 3:
                        fattore_tcs = 1.20
                    else:
                        fattore_tcs = 0.70
                    
                    accelerazione_massima_consentita = 1.0 - (abs(angolo_sterzo) - 0.10) * fattore_tcs
                    accelerazione_massima_consentita = max(0.18, min(1.0, accelerazione_massima_consentita))
                    acceleratore = min(acceleratore, accelerazione_massima_consentita)


            # ==========================================================
            # SISTEMI DI SICUREZZA ATTIVA E CORREZIONE TRAIETTORIA
            # ==========================================================
            if contatore_passi >= 80:
                # Correzione laterale a due livelli (versione stabile 1:42).
                # Interviene su entrambe le fasi (curva e rettilineo) per garantire
                # che l'auto non esca mai di pista per deriva laterale progressiva.
                if abs(posizione_tracciato) > 0.55:
                    if abs(posizione_tracciato) > 0.80:
                        correzione_sterzata = (abs(posizione_tracciato) - 0.80) * 2.2 + (0.80 - 0.55) * 0.8
                    else:
                        correzione_sterzata = (abs(posizione_tracciato) - 0.55) * 0.8
                    if posizione_tracciato > 0:
                        angolo_sterzo -= correzione_sterzata
                    else:
                        angolo_sterzo += correzione_sterzata

            # --- ALGORITMO DI RECUPERO (RECOVERY CONTROLLER STATE MACHINE) ---
            # Rileva se il veicolo è andato oltre i limiti della pista (|posizione_tracciato| > 1.0)
            if abs(posizione_tracciato) > 1.0:
                contatore_fuoripista += 1
            else:
                contatore_fuoripista = 0
                # Se eravamo in modalità di recupero e siamo rientrati in prossimità del centro tracciato, ripristiniamo la modalità standard
                if modalita_recupero is not None and abs(posizione_tracciato) < 0.5:
                    print(f"  [RECOVERY] Stato ripristinato: veicolo rientrato nei parametri di pista.")
                    modalita_recupero = None

            if contatore_fuoripista > 3 and modalita_recupero is None:
                modalita_recupero = "FORWARD_RECOVERY"
                passo_inizio_recupero = contatore_passi
                direzione_recupero_precedente = 1 if posizione_tracciato < 0 else -1
                print(f"  [RECOVERY] Rilevamento fuoripista. Stato: FORWARD_RECOVERY (posizione={posizione_tracciato:+.2f})")

            if (modalita_recupero == "FORWARD_RECOVERY" and
                contatore_passi - passo_inizio_recupero > 50 and
                abs(velocita_attuale) < 15.0):
                modalita_recupero = "REVERSE"
                passo_inizio_recupero = contatore_passi
                direzione_recupero_precedente = -direzione_recupero_precedente
                print(f"  [RECOVERY] Veicolo bloccato. Transizione stato: REVERSE (Retromarcia)")

            if (modalita_recupero == "REVERSE" and
                contatore_passi - passo_inizio_recupero > 60):
                modalita_recupero = "FORWARD_RECOVERY"
                passo_inizio_recupero = contatore_passi
                direzione_recupero_precedente = 1 if posizione_tracciato < 0 else -1
                print(f"  [RECOVERY] Fine ciclo retromarcia. Transizione stato: FORWARD_RECOVERY (Avanti)")

            if modalita_recupero == "FORWARD_RECOVERY":
                if velocita_attuale > 25.0:
                    # Fuori pista ad alta velocità: frenata di stabilizzazione ad assetto neutro (ruote allineate)
                    angolo_sterzo = 0.0
                    acceleratore = 0.0
                    freno = 0.8
                    marcia_attuale = max(1, marcia_attuale - 1)
                else:
                    # Rientro morbido sterzando verso il centro a velocità controllata
                    angolo_sterzo = -0.35 if posizione_tracciato > 0 else 0.35
                    acceleratore = 0.25
                    freno = 0.0
                    marcia_attuale = 1 if velocita_attuale < 18.0 else 2
            elif modalita_recupero == "REVERSE":
                # Retromarcia direzionata per orientare correttamente il muso dell'auto
                angolo_sterzo = direzione_recupero_precedente * 0.3
                acceleratore = 0.4
                freno = 0.0
                marcia_attuale = -1
            else:
                # --- SISTEMA DI CAMBIO SEQUENZIALE SEMI-AUTOMATICO ---
                # Modulazione delle marce basata su giri motore (RPM) e velocità lineare
                giri_motore = sensori.get("rpm", 0)
                tempo_corrente = time.time()
                
                # 1. Cambio marcia basato sulla soglia di giri (con tempo di ricarica di 0.3 secondi per evitare oscillazioni rapide)
                if tempo_corrente - tempo_ultimo_cambio > 0.3:
                    # Passaggio alla marcia superiore (Upshift) vicino al limitatore (9500 RPM)
                    if giri_motore > 9500 and marcia_attuale < 6:
                        marcia_attuale += 1
                        tempo_ultimo_cambio = tempo_corrente
                    # Scalata progressiva (Downshift)
                    else:
                        # Margine adattivo per evitare bloccaggi in frenata dovuti al freno motore (compression locking)
                        margine_scalata = 800 if freno > 0.1 else 0
                        
                        if marcia_attuale == 6 and giri_motore < (6800 - margine_scalata):
                            marcia_attuale = 5
                            tempo_ultimo_cambio = tempo_corrente
                        elif marcia_attuale == 5 and giri_motore < (6300 - margine_scalata):
                            marcia_attuale = 4
                            tempo_ultimo_cambio = tempo_corrente
                        elif marcia_attuale == 4 and giri_motore < (5800 - margine_scalata):
                            marcia_attuale = 3
                            tempo_ultimo_cambio = tempo_corrente
                        elif marcia_attuale == 3 and giri_motore < (4300 - margine_scalata) and marcia_attuale > 2:
                            marcia_attuale = 2
                            tempo_ultimo_cambio = tempo_corrente
                
                # 2. Controllo di fallback basato su velocità assoluta (previene spegnimenti del motore o marce non adeguate a bassa velocità)
                if velocita_attuale < 15.0:
                    marcia_attuale = 1
                elif velocita_attuale < 45.0:
                    marcia_attuale = min(marcia_attuale, 2)
                elif velocita_attuale < 75.0:
                    marcia_attuale = min(marcia_attuale, 3)

            # --- FILTRO PASSO-BASSO DELLO STERZO ---
            # Versione stabile (1:42): smorzamento solo a bassa velocita.
            # Ad alta velocita nessun filtro per massima reattivita della rete.
            if velocita_attuale < 65.0:
                alpha = 0.25 if velocita_attuale < 42.0 else 0.40
                angolo_sterzo = (sterzata_precedente * (1 - alpha)) + (angolo_sterzo * alpha)
            sterzata_precedente = angolo_sterzo

            # --- INVIO PACCHETTO COMANDI AL SIMULATORE ---
            # Applichiamo clipping di sicurezza sullo sterzo nell'intervallo [-1.0, 1.0]
            client_torcs.R.d["steer"]  = max(-1.0, min(1.0, angolo_sterzo))
            client_torcs.R.d["accel"]  = acceleratore
            client_torcs.R.d["brake"]  = freno
            client_torcs.R.d["gear"]   = marcia_attuale
            client_torcs.R.d["clutch"] = 0.0
            client_torcs.R.d["meta"]   = 0
            client_torcs.respond_to_server()

            # Logging periodico a scopo di debug delle telemetrie principali
            if contatore_passi % 50 == 0:
                print(f"  [TELEMETRIA] step={contatore_passi:05d} | "
                      f"sterzo={angolo_sterzo:+.2f} acc={acceleratore:.2f} freno={freno:.2f} | "
                      f"marcia={marcia_attuale} vel={velocita_attuale:5.1f} rpm={sensori.get('rpm', 0):.0f} | "
                      f"pos_tracciato={posizione_tracciato:+.2f}")
            contatore_passi += 1

        except KeyboardInterrupt:
            print("\n[INFO] Interruzione manuale da tastiera rilevata. Uscita dal programma.")
            break
        except Exception as errore:
            print(f"[ERROR] Rilevata un'eccezione imprevista: {errore}")


if __name__ == "__main__":
    main()