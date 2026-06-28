"""
manual_control.py — Guida manuale TORCS con controller PS4
Registra i dati di guida in dataset.csv per l'addestramento
Gruppo 24 BitSteer
"""
import pygame
import snakeoil3_jm2 as snakeoil3
import time


# ─── CLASSE CONTROLLER PS4 ────────────────────────────────────────────────────
# Gestisce l'inizializzazione e la lettura degli input del gamepad PS4.
# Mappa gli assi analogici e i pulsanti ai comandi di guida TORCS.
class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        # Verifica che almeno un controller sia collegato al PC
        if pygame.joystick.get_count() == 0:
            print("\n[ERRORE] Nessun controller rilevato!")
            print("Collega il controller PS4 al PC, assicurati che sia acceso e riavvia lo script.\n")
            exit()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"\n[OK] Controller connesso: {self.joystick.get_name()}\n")

        # Stato corrente dei comandi di guida
        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear':  1
        }
        self.last_gear_change = time.time()

    def update(self, sensors):
        pygame.event.pump()

        # ─── SENSIBILITÀ STERZO ───────────────────────────────────────────────
        # Riduce l'escursione massima dello sterzo per una guida più precisa.
        # Abbassare il valore per uno sterzo più morbido (es. 0.4),
        # aumentare per uno sterzo più reattivo (es. 0.8).
        sensibilita_sterzo = 0.6

        # ─── STERZO — LEVETTA SINISTRA (Asse 0) ──────────────────────────────
        # Legge l'asse orizzontale della levetta sinistra e applica una zona
        # morta (deadzone) del 15% per evitare drift involontario.
        steer_input = -self.joystick.get_axis(0)
        if abs(steer_input) < 0.15:
            steer_input = 0.0
        self.state['steer'] = steer_input * sensibilita_sterzo

        # ─── ACCELERATORE — GRILLETTO R2 (Asse 5) ────────────────────────────
        # R2 restituisce valori da -1 (rilasciato) a +1 (premuto a fondo).
        # Normalizzato in [0, 1] per TORCS.
        r2_val = self.joystick.get_axis(5)
        self.state['accel'] = (r2_val + 1.0) / 2.0

        # ─── FRENO — GRILLETTO L2 (Asse 4) ───────────────────────────────────
        # Stessa logica di R2 — normalizzato in [0, 1].
        l2_val = self.joystick.get_axis(4)
        self.state['brake'] = (l2_val + 1.0) / 2.0

        # ─── MARCE — PULSANTI ✖ CROCE (su) e ◼ QUADRATO (giù) ───────────────
        # Cooldown di 0.3 secondi tra un cambio marcia e l'altro per evitare
        # cambi multipli involontari in rapida successione.
        now = time.time()
        if now - self.last_gear_change > 0.3:
            if self.joystick.get_button(0):   # ✖ Croce — scala su
                self.state['gear'] += 1
                self.last_gear_change = now
            elif self.joystick.get_button(2): # ◼ Quadrato — scala giù
                self.state['gear'] -= 1
                self.last_gear_change = now

        # ─── CLIP VALORI ─────────────────────────────────────────────────────
        # Garantisce che tutti i valori rimangano nei range accettati da TORCS.
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0,  min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0,  min(1.0, self.state['brake']))
        self.state['gear']  = max(-1,   min(6,   self.state['gear']))


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():

    # ─── CONNESSIONE A TORCS ──────────────────────────────────────────────────
    # Connessione al simulatore e inizializzazione del controller.
    client     = snakeoil3.Client(p=3001, vision=False)
    controller = GamepadController()
    client.get_servers_input()

    print("Modalità Controller PS4 Attiva")
    print("Levetta SX: Sterzo | R2: Accel | L2: Freno | ✖ Croce: Marcia Su | ◼ Quadrato: Scala")
    print("ATTENZIONE: Se l'auto accelera da sola, premi e rilascia L2/R2 una volta per calibrarli!")
    print("Premi Ctrl+C per fermare e salvare.\n")

    # ─── INIZIALIZZAZIONE CSV ─────────────────────────────────────────────────
    # Apre il file di log e scrive l'intestazione con tutte le colonne.
    # I dati salvati verranno usati come dataset per addestrare la rete neurale.
    track_headers = ",".join([f"track_{i}" for i in range(19)])
    log_csv = open("dataset.csv", "w")
    log_csv.write(
        f"time,steer,accel,brake,gear,"
        f"speedX,trackPos,angle,rpm,damage,"
        f"distFromStart,curLapTime,"
        f"{track_headers}\n"
    )

    t0   = time.time()
    step = 0

    try:
        while True:
            # ─── LOOP PRINCIPALE ──────────────────────────────────────────────
            # Ad ogni timestep: legge i sensori, aggiorna lo stato del
            # controller, invia i comandi a TORCS e registra tutto sul CSV.
            client.get_servers_input()
            S = client.S.d

            controller.update(S)
            a = controller.state

            # Log a console ogni 50 step per monitorare la guida
            if step % 50 == 0:
                print(f"steer={a['steer']:.2f}  accel={a['accel']:.2f}  "
                      f"brake={a['brake']:.2f}  gear={a['gear']}  "
                      f"speed={S.get('speedX', 0):.0f}")

            # ─── INVIO COMANDI A TORCS ────────────────────────────────────────
            client.R.d['steer']  = a['steer']
            client.R.d['accel']  = a['accel']
            client.R.d['brake']  = a['brake']
            client.R.d['gear']   = a['gear']
            client.R.d['clutch'] = 0.0
            client.R.d['meta']   = 0
            client.respond_to_server()

            # ─── REGISTRAZIONE DATI SUL CSV ───────────────────────────────────
            # Salva timestamp, comandi e tutti i sensori del simulatore.
            # Questi dati costituiranno il dataset per il training.
            t      = time.time() - t0
            tracks = S.get('track', [200.0] * 19)
            if len(tracks) != 19:
                tracks = [200.0] * 19
            track_str = ",".join(f"{x:.4f}" for x in tracks)

            log_csv.write(
                f"{t:.4f},{a['steer']:.4f},{a['accel']:.4f},{a['brake']:.4f},{a['gear']},"
                f"{S.get('speedX', 0):.4f},{S.get('trackPos', 0):.6f},"
                f"{S.get('angle', 0):.6f},{S.get('rpm', 0):.2f},"
                f"{S.get('damage', 0):.1f},"
                f"{S.get('distFromStart', 0):.3f},{S.get('curLapTime', 0):.4f},"
                f"{track_str}\n"
            )

            step += 1

    except KeyboardInterrupt:
        # ─── SALVATAGGIO E CHIUSURA ───────────────────────────────────────────
        # Alla pressione di Ctrl+C flush e chiude il file CSV,
        # poi stampa quanti frame sono stati registrati.
        log_csv.flush()
        log_csv.close()
        print(f"\nSalvato 'dataset.csv'  ({step} frame registrati)")


if __name__ == "__main__":
    main()