import os
# Fix OMP Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# 1. CONFIGURAZIONE E PARAMETRI
# ==========================================
# Percorsi file
script_dir = os.path.dirname(__file__)
DATASET_PATH = os.path.join(script_dir, 'dataset_potenziato.pt')
MODEL_SAVE_PATH = os.path.join(script_dir, 'modello.pth')

# Parametri training
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-4
STATE_DIM = 23  # 19 track + speedX + angle + trackPos + rpm
ACTION_DIM = 3  # Accel_Brake, Steer, Gear

# Pesi per l'addestramento delle singole azioni
WEIGHT_ACCEL_BRAKE = 4.0 # Più importanza a gas/freno
WEIGHT_STEER = 1.5
WEIGHT_GEAR = 1

# ==========================================
# 2. DATASET E PREPROCESAMENTO
# ==========================================
class TorcsBCDataset(Dataset):
    def __init__(self, pt_file):
        print(f"Caricamento dataset binario: {pt_file}")
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"File {pt_file} non trovato! Esegui prima convert_to_pt.py")
            
        data = torch.load(pt_file)
        self.states = data['states']
        self.actions = data['actions']
        print(f"Dataset caricato: {len(self.states)} record.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ==========================================
# 3. ARCHITETTURA MODELLO
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


        # Teste di output separate per ogni azione
        self.accel_brake_head = nn.Linear(64, 1)
        self.steer_head = nn.Linear(64, 1)
        self.gear_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared_network(x)
        
        # Applichiamo i vincoli degli output singolarmente per ogni testa
        accel_brake = torch.tanh(self.accel_brake_head(features)) # [-1, 1] (Accelerazione > 0, Freno < 0)
        steer = torch.tanh(self.steer_head(features))             # [-1, 1]
        gear = self.gear_head(features)                           # Nessun vincolo (lineare)
        
        return torch.cat([accel_brake, steer, gear], dim=1)

# ==========================================
# 4. LOOP DI ADDESTRAMENTO
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"In uso: {device}")

    # Caricamento dati
    dataset = TorcsBCDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Inizializzazione modello
    model = BCModel(STATE_DIM, ACTION_DIM).to(device)
    
    # Ottimizzatore con parameter groups per controllare l'apprendimento delle teste singolarmente
    optimizer = optim.Adam([
        {'params': model.shared_network.parameters(), 'lr': LEARNING_RATE},
        {'params': model.accel_brake_head.parameters(), 'lr': LEARNING_RATE * WEIGHT_ACCEL_BRAKE},
        {'params': model.steer_head.parameters(), 'lr': LEARNING_RATE * WEIGHT_STEER},
        {'params': model.gear_head.parameters(), 'lr': LEARNING_RATE * WEIGHT_GEAR}
    ])
    
    # Loss separate
    criterion_accel_brake = nn.MSELoss()
    criterion_steer = nn.MSELoss()
    criterion_gear = nn.MSELoss()

    print("Inizio addestramento Behavioral Cloning (Multi-Head)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        # total_loss_accel = 0
        # total_loss_steer = 0
        # total_loss_gear = 0
        for states, actions in dataloader:
            states, actions = states.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(states)
            
            # Divisione delle uscite
            pred_accel_brake = outputs[:, 0:1]
            pred_steer = outputs[:, 1:2]
            pred_gear = outputs[:, 2:3]
            
            target_accel_brake = actions[:, 0:1]
            target_steer = actions[:, 1:2]
            target_gear = actions[:, 2:3]
            
            # Calcolo loss singole
            loss_accel_brake = criterion_accel_brake(pred_accel_brake, target_accel_brake)
            loss_steer = criterion_steer(pred_steer, target_steer)
            loss_gear = criterion_gear(pred_gear, target_gear)
            
            # Loss totale pesata
            loss = (loss_accel_brake * WEIGHT_ACCEL_BRAKE) + \
                   (loss_steer * WEIGHT_STEER) + \
                   (loss_gear * WEIGHT_GEAR)
            
            loss.backward()
            optimizer.step()
            
            # Teniamo traccia della loss reale media (non pesata) per la stampa
            total_loss += (loss_accel_brake.item() + loss_steer.item() + loss_gear.item()) / 3.0
            # total_loss_accel += loss_accel_brake.item()
            # total_loss_steer += loss_steer.item()
            # total_loss_gear += loss_gear.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoca [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.6f}")
            # print(f"Epoca [{epoch+1}/{EPOCHS}], Loss Media: {total_loss/len(dataloader):.6f} | Accel/Brake: {total_loss_accel/len(dataloader):.6f} | Steer: {total_loss_steer/len(dataloader):.6f} | Gear: {total_loss_gear/len(dataloader):.6f}")

    # Salvataggio
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modello salvato in {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
