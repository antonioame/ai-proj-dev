"""
train.py — Addestramento rete neurale PyTorch
Dataset: solo dataset.csv
Lanciato una sola volta — salva model.pt e scaler.pkl
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ─── SEED PER RIPRODUCIBILITÀ ─────────────────────────────────────────────────
# Fissa i seed di numpy e PyTorch per garantire risultati identici
# ad ogni esecuzione del training.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── DEFINIZIONE FEATURE DI INPUT ────────────────────────────────────────────
# 29 feature totali: 19 sensori di distanza pista + velocità (3) +
# posizione laterale, angolo, RPM + velocità ruote (4).
track_cols = [f'track_{i}' for i in range(19)]
input_cols = (
    track_cols +
    ['speedX', 'speedY', 'speedZ', 'trackPos', 'angle', 'rpm',
     'wheelSpin_0', 'wheelSpin_1', 'wheelSpin_2', 'wheelSpin_3']
)

# ─── CARICAMENTO DATASET ─────────────────────────────────────────────────────
# Legge il file dataset.csv contenente le registrazioni di guida umana.
print("Caricamento dataset.csv ...")
df = pd.read_csv("dataset.csv")
print(f"  Righe totali: {len(df)}")

# ─── PREPROCESSING — TARGET STERZO ───────────────────────────────────────────
# Usa steer_raw (sterzo grezzo non scalato) se disponibile,
# altrimenti usa steer. Clipping in [-1, +1] per sicurezza.
if 'steer_raw' in df.columns:
    df['steer'] = df['steer_raw'].clip(-1, 1)
else:
    df['steer'] = df['steer'].clip(-1, 1)

# Colonne opzionali — imposta a zero se assenti nel dataset
for col in ['speedY', 'speedZ']:
    if col not in df.columns: df[col] = 0.0
for j in range(4):
    if f'wheelSpin_{j}' not in df.columns: df[f'wheelSpin_{j}'] = 0.0

# ─── FILTRAGGIO DATASET ───────────────────────────────────────────────────────
# Rimuove campioni di scarsa qualità per migliorare il training:
# - sensori di distanza nulli (veicolo fuori pista)
# - velocità troppo bassa (veicolo fermo o quasi)
# - posizione laterale eccessiva (fuori pista)
# - cambi di sterzo bruschi (zigzag nel dataset)
df = df[df[track_cols].min(axis=1) > 0]
df = df[df['speedX'] > 5]
df = df[df['trackPos'].abs() < 0.9]
df = df.dropna(subset=input_cols + ['steer', 'accel', 'brake'])
df = df.sort_values('distFromStart').reset_index(drop=True)
df['steer_diff'] = df['steer'].diff().abs()
df = df[df['steer_diff'] < 0.3].drop(columns='steer_diff')
print(f"Campioni dopo pulizia: {len(df)}")

# ─── PREPARAZIONE FEATURE E TARGET ───────────────────────────────────────────
# X: matrice delle feature di input (29 colonne)
# Y: matrice dei target continui (steer, accel, brake)
# gears: vettore delle marce come classi intere per la CrossEntropy
X = df[input_cols].values.astype(np.float32)
Y = df[['steer', 'accel', 'brake']].values.astype(np.float32)

gear_offset = int(df['gear'].min())
gears   = (df['gear'] - gear_offset).values.astype(np.int64)
n_gears = int(df['gear'].max()) - gear_offset + 1

# ─── NORMALIZZAZIONE Z-SCORE ──────────────────────────────────────────────────
# Standardizza le feature sottraendo la media e dividendo per la deviazione
# standard. I parametri vengono salvati in scaler.pkl per essere riutilizzati
# in fase di inferenza in driver.py.
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8
X_s  = (X - mean) / std

joblib.dump({
    'mean': mean, 'std': std,
    'input_cols': input_cols,
    'gear_offset': gear_offset,
    'n_gears': n_gears,
}, "scaler.pkl")
print("Scaler salvato.")

# ─── SPLIT TRAIN / VALIDATION ────────────────────────────────────────────────
# 90% dei campioni per il training, 10% per la validation.
# Lo split è stratificato per seed per garantire riproducibilità.
X_train, X_val, Y_train, Y_val, G_train, G_val = train_test_split(
    X_s, Y, gears, test_size=0.1, random_state=SEED
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

# ─── DATALOADER ───────────────────────────────────────────────────────────────
# Crea i dataset PyTorch e i DataLoader per il training e la validation.
# Il training shuffle i dati ad ogni epoca per evitare overfitting sull'ordine.
train_ds = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(Y_train),
    torch.from_numpy(G_train),
)
val_ds = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(Y_val),
    torch.from_numpy(G_val),
)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)


# ─── ARCHITETTURA RETE NEURALE ────────────────────────────────────────────────
# MLP multi-task con backbone condiviso e tre teste di output dedicate.
# Identica alla classe in driver.py — necessario che le architetture
# coincidano per caricare correttamente i pesi salvati.
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


# ─── INIZIALIZZAZIONE MODELLO E OTTIMIZZATORE ─────────────────────────────────
# Usa GPU se disponibile, altrimenti CPU.
# Adam con lr=1e-3 e ReduceLROnPlateau che dimezza il learning rate
# se la val_loss non migliora per 5 epoche consecutive.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = DrivingNet(in_dim=len(input_cols), n_gears=n_gears).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
mse = nn.MSELoss()
ce  = nn.CrossEntropyLoss()

best_val   = float('inf')
patience   = 15
no_improve = 0

# ─── LOOP DI TRAINING ─────────────────────────────────────────────────────────
# Per ogni epoca: forward pass, calcolo loss pesata, backpropagation.
# Loss pesata: sterzo ×2.0 (controllo più critico), freno ×1.5,
# acceleratore ×1.0, marcia ×0.5 (gestita anche da logiche euristiche).
# Early stopping: interrompe se val_loss non migliora per 15 epoche.
# Salva il modello con la migliore val_loss in model.pt.
print("\nAddestramento...")
for epoch in range(150):
    model.train()
    for xb, yb, gb in train_loader:
        xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
        steer_p, ab_p, gear_p = model(xb)
        loss = (
            2.0 * mse(steer_p.squeeze(), yb[:, 0]) +
            1.0 * mse(ab_p[:, 0],        yb[:, 1]) +
            1.5 * mse(ab_p[:, 1],        yb[:, 2]) +
            0.5 * ce(gear_p,             gb)
        )
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # ─── VALIDATION ───────────────────────────────────────────────────────────
    # Calcola la val_loss sull'intero validation set senza aggiornare i pesi.
    # Aggiorna lo scheduler e controlla l'early stopping.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb, gb in val_loader:
            xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
            steer_p, ab_p, gear_p = model(xb)
            val_loss += (
                2.0 * mse(steer_p.squeeze(), yb[:, 0]) +
                1.0 * mse(ab_p[:, 0],        yb[:, 1]) +
                1.5 * mse(ab_p[:, 1],        yb[:, 2]) +
                0.5 * ce(gear_p,             gb)
            ).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1:3d} — val_loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "model.pt")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping a epoch {epoch+1}")
            break

print(f"\nBest val_loss: {best_val:.4f}")
print("Salvato: model.pt + scaler.pkl")
print("Ora puoi lanciare driver.py!")