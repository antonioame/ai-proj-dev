import os
import numpy as np
import snakeoil3_jm2 as snakeoil3

# Forza la compatibilità OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class TorcsEnv:
    def __init__(self, port=3001, host="localhost"):
        self.port, self.host = port, host
        self.client = None
        
        # --- VARIABILI DI STATO ---
        self.step_count = 0
        self.stall_counter = 0

    def _get_obs(self, sensors):
        raw_track = np.array(sensors['track'])
        raw_track[raw_track == -1.0] = 200.0
        track_sensors = raw_track / 200.0
        speed_x = np.array([sensors['speedX']]) / 340.0
        angle = np.array([sensors['angle']]) / np.pi
        track_pos = np.clip(np.array([sensors['trackPos']]), -2.0, 2.0)
        rpm = np.array([sensors['rpm']]) / 10000.0
        return np.concatenate((track_sensors, speed_x, angle, track_pos, rpm)).astype(np.float32)

    def reset(self):
        self.step_count = 0
        self.stall_counter = 0
        
        if self.client: 
            self.client.R.d['meta'] = 1
            self.client.respond_to_server()
        
        # Connessione al server
        print(f"\n[ENV] Reset: Connessione a TORCS (Porta {self.port})...")
        self.client = snakeoil3.Client(p=self.port, H=self.host)
        self.client.get_servers_input()
        return self._get_obs(self.client.S.d)

    def step(self, action):
        self.step_count += 1
        
        # Clipping per sicurezza
        steer = np.clip(float(action[0]), -1.0, 1.0)
        accel = np.clip(float(action[1]), 0.0, 1.0)
        brake = np.clip(float(action[2]), 0.0, 1.0)

        # ==========================================
        # 1. LOGICA COMANDI (Partenza & Cambio)
        # ==========================================
        s_prev = self.client.S.d
        cur_lap_time = s_prev.get('curLapTime', -1.0)
        speed_prev = s_prev.get('speedX', 0.0)
        
        # Forza la partenza solo per i primi 40 step se l'auto è ferma
        if cur_lap_time <= 0.1 and self.step_count < 40:
            self.client.R.d['accel'] = 1.0
            self.client.R.d['brake'] = 0.0
            self.client.R.d['gear'] = 1
            self.client.R.d['steer'] = steer
            self.client.R.d['clutch'] = 0.0
        else:
            self.client.R.d['steer'] = steer
            self.client.R.d['accel'] = accel
            self.client.R.d['brake'] = brake
            self.client.R.d['clutch'] = 0.0
            
            rpm = s_prev.get('rpm', 0)
            gear = s_prev.get('gear', 0)
            if gear <= 0: self.client.R.d['gear'] = 1
            elif rpm > 9000: self.client.R.d['gear'] = min(gear + 1, 6)
            elif rpm < 3500 and gear > 1: self.client.R.d['gear'] = gear - 1
            else: self.client.R.d['gear'] = gear

        self.client.respond_to_server()
        self.client.get_servers_input()
        
        s = self.client.S.d
        if not s: return np.zeros(23, dtype=np.float32), True

        obs = self._get_obs(s)
        
        # ==========================================
        # 2. GESTIONE FINE EPISODIO (Stallo)
        # ==========================================
        speed = s.get('speedX', 0)

        # Terminate se fuori pista o stallo
        terminated = False
            
        if speed < 2.0 and cur_lap_time > 0.5:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        if self.stall_counter > 200:
            terminated = True

        return obs, terminated

    def close(self):
        if self.client:
            self.client.R.d['meta'] = 1
            self.client.respond_to_server()
