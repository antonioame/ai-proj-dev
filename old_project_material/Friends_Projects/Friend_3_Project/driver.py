"""
driver.py — Guida autonoma TORCS — PyTorch
Imitation Learning su pista Corkscrew — Gruppo 24 BitSteer
"""
import time
import numpy as np
import joblib
import torch
import torch.nn as nn
import snakeoil3_jm2 as snakeoil3

# ─── ARCHITETTURA RETE NEURALE ───────────────────────────────────────────────
    # MLP multi-task con backbone condiviso e tre teste di output:
    # - head_steer: predice lo sterzo (Tanh → [-1, +1])
    # - head_accel_brake: predice acceleratore e freno (Sigmoid → [0, 1])
    # - head_gear: predice la marcia (argmax su N classi)
class DrivingNet(nn.Module):
    def __init__(self, in_dim, n_gears=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128),    nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),     nn.ReLU(),
        )
        self.head_steer       = nn.Linear(64, 1)
        self.head_accel_brake = nn.Linear(64, 2)
        self.head_gear        = nn.Linear(64, n_gears)

    def forward(self, x):
        h = self.backbone(x)
        return (
            torch.tanh(self.head_steer(h)),
            torch.sigmoid(self.head_accel_brake(h)),
            self.head_gear(h),
        )

# ─── CARICAMENTO MODELLO E SCALER ────────────────────────────────────────
    # Carica i pesi della rete (model.pt) e i parametri di normalizzazione
    # (scaler.pkl) salvati durante il training.
def main():
    print("Inizializzazione modello di guida autonoma...")
    scaler      = joblib.load("scaler.pkl")
    mean        = scaler["mean"]
    std         = scaler["std"]
    input_cols  = scaler["input_cols"]
    gear_offset = scaler["gear_offset"]
    n_gears     = scaler.get("n_gears", 6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DrivingNet(in_dim=len(input_cols), n_gears=n_gears).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()
    print(f"Modello caricato correttamente — dispositivo: {device}")

    # ─── CONNESSIONE A TORCS ─────────────────────────────────────────────────
    # Connessione al simulatore via socket UDP sulla porta 3001.
    # get_servers_input() attende che TORCS sia pronto prima di iniziare.
    client = snakeoil3.Client(p=3001, vision=False)
    client.get_servers_input()
    print("Connessione a TORCS stabilita. Avvio sessione di guida autonoma...\n")

    # ─── VARIABILI DI STATO ──────────────────────────────────────────────────
    # Variabili di controllo del loop di guida: gestione marce, rilevamento
    # congelamento, recovery da fuoripista, contatore giri e curve.
    last_gear_change     = 0.0
    last_gear            = 1
    step                 = 0
    last_positions       = []
    last_speeds          = []
    off_track_counter    = 0
    recovery_mode        = None
    recovery_start_step  = 0
    last_recovery_dir    = 0
    last_lap_time        = 0.0
    current_lap          = 0
    lap_start_step       = 0
    first_curve_active   = False
    first_curve_direction = 0
    last_curve_active    = False
    last_smooth_steer = 0.0
    last_curve_direction = 0

    while True:
        try:
            client.get_servers_input()
            S = client.S.d

            # ─── RILEVAMENTO FINE GIRO ───────────────────────────────────────
            # curLapTime si azzera a ogni nuovo giro. Quando scende bruscamente
            # significa che un giro è stato completato — stampa il tempo.
            cur_lap_time = S.get("curLapTime", 0.0)
            if cur_lap_time < last_lap_time - 1.0:
                current_lap += 1
                elapsed = step - lap_start_step
                print(f"\nGiro {current_lap} completato — tempo: ~{last_lap_time:.1f}s ({elapsed} step)\n")
                lap_start_step = step
            last_lap_time = cur_lap_time

            # ─── LETTURA SENSORI ─────────────────────────────────────────────
            # Legge i sensori dal simulatore e li organizza in un dizionario
            # con gli stessi nomi usati durante il training.
            track  = S.get("track", [200.0]*19)
            wheels = S.get("wheelSpinVel", [0.0]*4)
            if len(track)  != 19: track  = [200.0]*19
            if len(wheels) != 4:  wheels = [0.0]*4

            vals = {}
            for i in range(19): vals[f'track_{i}']     = float(track[i])
            for i in range(4):  vals[f'wheelSpin_{i}'] = float(wheels[i])
            vals['speedX']   = float(S.get('speedX',   0))
            vals['speedY']   = float(S.get('speedY',   0))
            vals['speedZ']   = float(S.get('speedZ',   0))
            vals['trackPos'] = float(S.get('trackPos', 0))
            vals['angle']    = float(S.get('angle',    0))
            vals['rpm']      = float(S.get('rpm',      0))

            # ─── INFERENZA RETE NEURALE ──────────────────────────────────────
            # Normalizza le feature con z-score (stessi parametri del training),
            # esegue l'inferenza e ottiene sterzo, accelerazione e freno.
            x  = np.array([vals[c] for c in input_cols], dtype=np.float32)
            xn = (x - mean) / std
            xt = torch.from_numpy(xn).unsqueeze(0).to(device)

            with torch.no_grad():
                steer_t, ab_t, _ = model(xt)

            steer = float(steer_t.item())
            accel = float(ab_t[0, 0].item())
            brake = float(ab_t[0, 1].item())

            current_speed = S.get("speedX", 0)
            track_arr     = S.get("track", [200]*19)
            trackPos      = S.get("trackPos", 0)
            rpm           = S.get("rpm", 0)
            dist          = float(S.get("distFromStart", 0))

            # ─── WATCHDOG ANTI-FREEZE ────────────────────────────────────────
            # Se posizione e velocità rimangono identiche per 50 step consecutivi
            # il simulatore è congelato — disconnette il client.
            if step > 100:
                last_positions.append(trackPos)
                last_speeds.append(current_speed)
                if len(last_positions) > 50:
                    last_positions.pop(0)
                    last_speeds.pop(0)
                    if len(set(last_positions)) == 1 and len(set(last_speeds)) == 1:
                        print("Simulatore inattivo — disconnessione in corso.")
                        break

            # ─── FASE DI PARTENZA ────────────────────────────────────────────
            # Nei primi 80 step forza gas a fondo e riduce lo sterzo al 50%
            # per evitare sterzate brusche durante l'accelerazione iniziale
            if step < 80:
                accel = 1.0; brake = 0.0
                steer = steer * 0.5
                if   current_speed < 5:  last_gear = 1
                elif current_speed < 15: last_gear = 2
                else:                    last_gear = 3
            else:
                # ─── CRUISE CONTROL LONGITUDINALE ────────────────────────────
                # La velocità target dipende dal lookahead — il massimo dei 9
                # sensori centrali del ventaglio — che stima la curvatura
                # della strada davanti al veicolo.
                track_lookahead = max(track_arr[5:14])

                if   track_lookahead > 170: target_speed = 210.0
                elif track_lookahead > 120: target_speed = 208.0
                elif track_lookahead > 80:  target_speed = 160.0
                elif track_lookahead > 50:  target_speed = 126.0
                elif track_lookahead > 30:  target_speed = 80.0
                else:                       target_speed = 55.0

                # Riduzione velocità target se il veicolo è fuori centro pista
                if abs(trackPos) > 0.65: target_speed = min(target_speed, 98.0)
                if abs(trackPos) > 0.78: target_speed = min(target_speed, 61.0)

                required_brake_distance = (current_speed ** 2) / 232.0 + 5.0

                if current_speed < target_speed:
                    diff  = target_speed - current_speed
                    accel = 1.0 if track_lookahead > 120 else min(1.0, diff / 8.0)
                    brake = 0.0
                else:
                    # ─── ABS DINAMICO ─────────────────────────────────────────
                    # Limita il freno massimo in funzione della velocità per
                    # evitare bloccaggi e perdita di direzionalità in frenata.
                    diff  = current_speed - target_speed
                    accel = 0.0
                    braking_lookahead = min(track_lookahead, track_arr[9] * 1.05) if current_speed > 160 else track_lookahead
                    if braking_lookahead < required_brake_distance:
                        if   current_speed > 140: max_brake = 0.59
                        elif current_speed > 90:  max_brake = 0.71
                        else:                     max_brake = 0.84
                        if abs(steer) > 0.08:
                            max_brake = max(0.38, min(max_brake, max_brake - (abs(steer) - 0.08) * 0.75))
                        brake = min(max_brake, diff / 10.0)
                    else:
                        brake = 0.0

                # ─── TCS — TRACTION CONTROL SYSTEM ───────────────────────────
                # Riduce l'accelerazione quando si sterza, per prevenire
                # sovrasterzi di potenza in curva. Il fattore dipende dalla
                # marcia: più aggressivo nelle marce basse (coppia elevata).
                if abs(steer) > 0.10:
                    if   last_gear <= 2: factor = 1.45
                    elif last_gear == 3: factor = 1.20
                    else:               factor = 0.70
                    max_accel = max(0.18, min(1.0, 1.0 - (abs(steer) - 0.10) * factor))
                    accel     = min(accel, max_accel)

                # ─── OVERRIDE ZONE SPECIALI (accel/brake) ────────────────────
                # In alcune zone il cruise control generale produce comportamenti
                # non ottimali. Questi override forzano accel e brake corretti
                # basandosi sulla distanza percorsa dall'inizio pista.
                if 3305 < dist < 3608:
                    # Ultimo rettilineo: gas a fondo fino al traguardo
                    brake = 0.0
                    accel = 1.0
                    last_gear = min(last_gear + 1, 6) if rpm > 7000 else last_gear


                if 117 < dist < 221:
                    # Rettilineo iniziale: evita frenata inutile
                    brake = 0.0
                    accel = max(accel, 0.4)

                if 983 < dist < 1075:
                    # Curva affrontabile senza freno
                    brake = 0.0
                    accel = max(accel, 0.3)

                if 2650 < dist < 2784:
                    # Curva sinistra: freno dosato per non perdere velocità
                    brake = min(brake, 0.25)
                    accel = max(accel, 0.2)


                # ─── OVERRIDE ZONE CRITICHE (sterzo) ─────────────────────────
                # Il modello tende a sterzare eccessivamente in queste zone,
                # producendo una doppia sterzata che porta fuori pista.
                # Ogni zona è stata identificata tramite analisi dei log di
                # telemetria e corretta con clamp e forzature mirate.

                # Prima curva — clamp sterzo e unica sterzata netta
                if 393 < dist < 537:
                    if not first_curve_active:
                        first_curve_active = True
                        if steer != 0.0:
                            first_curve_direction = int(np.sign(steer))
                        elif trackPos != 0.0:
                            first_curve_direction = int(-np.sign(trackPos))
                        else:
                            first_curve_direction = 1

                    brake = min(brake, 0.18)
                    accel = max(accel, 0.45)
                    if current_speed < 85 and abs(trackPos) < 0.55:
                        accel = max(accel, 0.65)

                    max_first_curve_steer = 0.28 if abs(trackPos) < 0.45 else 0.22
                    if np.sign(steer) == first_curve_direction:
                        steer = np.sign(steer) * min(abs(steer), max_first_curve_steer)
                    else:
                        steer = np.sign(first_curve_direction) * (max_first_curve_steer * 0.65)

                    if abs(trackPos) > 0.50:
                        accel = min(accel, 0.35)
                        brake = max(brake, 0.08)

                else:
                    first_curve_active = False

                # Curva 1876-1960 — stessa logica della prima curva
                if 1876 < dist < 1960:
                    brake = min(brake, 0.18)
                    accel = max(accel, 0.45)
                    steer = max(-0.28, min(0.28, steer))
                    if abs(trackPos) > 0.50:
                        accel = min(accel, 0.35)
                        brake = max(brake, 0.08)

                # Tornante 2400-2530 — rischio uscita destra
                if 2400 < dist < 2530:
                    steer = max(-0.32, min(0.32, steer))
                    if abs(trackPos) > 0.35 and np.sign(steer) == np.sign(trackPos):
                        steer = np.sign(steer) * 0.24
                        accel = min(accel, 0.35)
                        brake = max(brake, 0.08)

                # Ultima curva 3230-3320 — rischio uscita sinistra
                if 3230 < dist < 3320:
                    steer = max(-0.20, min(0.30, steer))
                    if trackPos < -0.40:
                        steer = +0.25

                # Curva 1453-1560 — rischio uscita destra
                if 1453 < dist < 1560:
                    steer = max(-0.28, min(0.28, steer))
                    if abs(trackPos) > 0.50:
                        accel = min(accel, 0.35)
                        brake = max(brake, 0.08)
                        if trackPos > 0.50:
                            steer = -0.22
                        elif trackPos < -0.50:
                            steer = +0.22

            # ─── CORREZIONE LATERALE ─────────────────────────────────────────
            # Se il veicolo si avvicina troppo al bordo pista (|trackPos| > 0.72)
            # applica una forza di richiamo verso il centro proporzionale
            # allo scostamento. Identica alla logica usata dal modello di riferimento.
            if step >= 80 and abs(trackPos) > 0.72:
                correction = (abs(trackPos) - 0.72) * 1.1
                steer -= correction if trackPos > 0.72 else -correction

            # ─── GESTIONE FUORIPISTA E RECOVERY ─────────────────────────────
            # Conta i timestep consecutivi fuori pista. Dopo 5 step attiva
            # la modalità FORWARD per rientrare. Se il veicolo si blocca
            # (velocità < 15 km/h per 50 step) passa in REVERSE.
            if abs(trackPos) > 1.0:
                off_track_counter += 1
            else:
                off_track_counter = 0
                if recovery_mode and abs(trackPos) < 0.5:
                    print("Rientro in pista completato.")
                    recovery_mode = None

            if off_track_counter > 5 and recovery_mode is None:
                recovery_mode       = "FORWARD"
                recovery_start_step = step
                last_recovery_dir   = 1 if trackPos < 0 else -1
                print(f"Fuoripista rilevato — avvio procedura di rientro (pos={trackPos:+.2f})")

            if recovery_mode == "FORWARD" and step - recovery_start_step > 50 and abs(current_speed) < 15:
                recovery_mode       = "REVERSE"
                recovery_start_step = step
                last_recovery_dir   = -last_recovery_dir
                print("Veicolo bloccato — attivazione retromarcia.")

            if recovery_mode == "REVERSE" and step - recovery_start_step > 60:
                recovery_mode       = "FORWARD"
                recovery_start_step = step
                last_recovery_dir   = 1 if trackPos < 0 else -1

            if recovery_mode == "FORWARD":
                if current_speed > 25:
                    steer = 0.0; accel = 0.0; brake = 0.8
                    last_gear = max(1, last_gear - 1)
                else:
                    steer = -0.35 if trackPos > 0 else 0.35
                    accel = 0.25; brake = 0.0; last_gear = 1
            elif recovery_mode == "REVERSE":
                steer = last_recovery_dir * 0.3
                accel = 0.4; brake = 0.0; last_gear = -1
            else:
                # ─── GESTIONE MARCE ───────────────────────────────────────────
                # Cambia marcia in su quando gli RPM superano 9500.
                # Scala in giù in base agli RPM con isteresi quando si frena.
                # Forza marce basse a basse velocità per evitare spegnimento.
                now = time.time()
                if now - last_gear_change > 0.3:
                    if rpm > 9500 and last_gear < 6:
                        last_gear += 1; last_gear_change = now
                    else:
                        ds = 800 if brake > 0.1 else 0
                        if   last_gear == 6 and rpm < 6800 - ds: last_gear = 5; last_gear_change = now
                        elif last_gear == 5 and rpm < 6300 - ds: last_gear = 4; last_gear_change = now
                        elif last_gear == 4 and rpm < 5800 - ds: last_gear = 3; last_gear_change = now
                        elif last_gear == 3 and rpm < 4300 - ds: last_gear = 2; last_gear_change = now

                if   current_speed < 15: last_gear = 1
                elif current_speed < 45: last_gear = min(last_gear, 2)
                elif current_speed < 75: last_gear = min(last_gear, 3)

            # ─── INVIO COMANDI AL SIMULATORE ─────────────────────────────────
            # Invia sterzo, accel, freno e marcia a TORCS. Clip su [−1,+1]
            # e [0,1] per sicurezza prima dell'invio.
            client.R.d["steer"]  = max(-1.0, min(1.0, steer))
            client.R.d["accel"]  = float(np.clip(accel, 0.0, 1.0))
            client.R.d["brake"]  = float(np.clip(brake, 0.0, 1.0))
            client.R.d["gear"]   = last_gear
            client.R.d["clutch"] = 0.0
            client.R.d["meta"]   = 0
            client.respond_to_server()

            if step % 50 == 0:
                print(f"step={step:05d} | steer={steer:+.2f} acc={accel:.2f} brk={brake:.2f} "
                      f"g={last_gear} v={current_speed:5.1f} tp={trackPos:+.2f} dist={dist:.0f}m")
            step += 1

        except KeyboardInterrupt:
            print("\nSessione interrotta dall'utente."); break
        except Exception as e:
            print(f"Errore durante l'esecuzione: {e}")


if __name__ == "__main__":
    main()