import os
# Fix OMP Error #15 - Deve essere impostato PRIMA degli altri import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import numpy as np

def convert_to_pt(json_file, pt_file):
    print(f"Caricamento {json_file}...")
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    states = []
    actions = []
    
    print(f"Conversione di {len(raw_data)} record...")
    for row in raw_data:
        # --- NORMALIZZAZIONE STATO (Coerente con train_bc_prova.py) ---
        track = []
        for t in row['track']:
            val = 200.0 if t == -1.0 else t
            track.append(min(1.0, val / 200.0))
        
        speed_x = row['speedX'] / 340.0
        angle = row['angle'] / np.pi
        track_pos = np.clip(row['trackPos'], -2.0, 2.0)
        rpm = row['rpm'] / 10000.0
        
        state = track + [speed_x, angle, track_pos, rpm]
        
        # --- AZIONI ---
        steer = row.get('action_steer', 0.0)
        accel = row.get('action_accel', 0.0)
        brake = row.get('action_brake', 0.0)
        gear = float(row.get('action_gear', 1.0))
        
        # Accelerazione e freno uniti: -1 a 1
        accel_brake = accel - brake
        
        # Ordine uscite: accelerazione/freno, sterzo, marcia
        action = [accel_brake, steer, gear]
        
        states.append(state)
        actions.append(action)
    
    print(f"Salvataggio in {pt_file}...")
    torch.save({
        'states': torch.tensor(states, dtype=torch.float32),
        'actions': torch.tensor(actions, dtype=torch.float32)
    }, pt_file)
    print("Conversione completata!")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    json_f = os.path.join(script_dir, 'dataset_potenziato.json')
    pt_f = os.path.join(script_dir, 'dataset_potenziato.pt')
    convert_to_pt(json_f, pt_f)
