import os
import sys

# Forza la compatibilità OpenMP (Risolve Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Aggiungiamo la cartella superiore al path per importare TorcsEnv e snakeoil
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Forziamo l'uso di snakeoil3_gym (associata alla porta 3001) invece di snakeoil3_jm2
try:
    import snakeoil3_gym as snakeoil3
    sys.modules['snakeoil3_jm2'] = snakeoil3
except ImportError:
    pass

import torch
import torch.nn as nn
import numpy as np
import time

try:
    from torcs_env import TorcsEnv
except ImportError as e:
    print(f"Dettaglio Errore Importazione: {e}")
    print("Assicurati che torcs_env.py sia nella cartella superiore e che le sue dipendenze siano installate.")
    sys.exit(1)

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modello.pth')
PORT = 3001
STATE_DIM = 23
ACTION_DIM = 3

# ==========================================
# 2. DEFINIZIONE ARCHITETTURA (Deve coincidere con train_bc_prova.py)
# ==========================================
class BCModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(BCModel, self).__init__()
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.accel_brake_head = nn.Linear(64, 1)
        self.steer_head = nn.Linear(64, 1)
        self.gear_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared_network(x)
        accel_brake = torch.tanh(self.accel_brake_head(features))
        steer = torch.tanh(self.steer_head(features))
        gear = self.gear_head(features)
        return torch.cat([accel_brake, steer, gear], dim=1)

# ==========================================
# 3. TEST LOOP
# ==========================================
def test_bc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Caricamento modello
    model = BCModel(STATE_DIM, ACTION_DIM).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Errore: Modello {MODEL_PATH} non trovato. Esegui prima l'addestramento.")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Modello BC caricato da {MODEL_PATH}")

    # Inizializzazione ambiente
    env = TorcsEnv(port=PORT)
    
    try:
        while True:
            obs = env.reset()
            done = False
            
            print("\nInizio nuovo episodio...")
            while not done:
                # L'osservazione restituita da torcs_env.py è già normalizzata
                # Conversione in tensore
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action_t = model(obs_t)
                
                model_out = action_t.squeeze(0).cpu().numpy()
                accel_brake = float(model_out[0])
                steer = float(model_out[1])
                gear = int(round(float(model_out[2])))
                
                # Decodifica di accel_brake in acceleratore e freno
                if accel_brake > 0:
                    accel = accel_brake
                    brake = 0.0
                else:
                    accel = 0.0
                    brake = -accel_brake
                
                # Ricostruzione dell'azione per l'ambiente (es. [steer, accel, brake])
                # Nota: se l'ambiente supporta la marcia in input, aggiungi `gear` all'array.
                action = np.array([steer, accel, brake])
                
                # Step nell'ambiente
                obs, done = env.step(action)
                
                # Debug info (opzionale)
                # print(f"Action: S:{action[0]:.2f} A:{action[1]:.2f} B:{action[2]:.2f}", end='\r')
            
            print(f"Episodio terminato.")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTest interrotto dall'utente.")
    finally:
        env.close()

if __name__ == "__main__":
    test_bc()
