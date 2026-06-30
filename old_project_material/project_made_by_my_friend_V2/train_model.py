"""
Progetto di Intelligenza Artificiale - Pipeline di Addestramento Modello di Guida
---------------------------------------------------------------------------------
Descrizione: Script per l'addestramento supervisionato del modello MLP DrivingNet.
Prepara il dataset di telemetria proveniente da TORCS, normalizza i dati di input,
addestra la rete neurale mediante un approccio Multi-Task Learning e salva i pesi ottimali.
"""

import sys
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────
# 1. CONFIGURAZIONE E ARCHIVIO DELLE FEATURE (INPUT SELECTION)
# ──────────────────────────────────────────────────────────────────────────
# 19 sensori di distanza disposti a raggiera sul fronte del veicolo
COLONNE_TRACCIATO = [f"track_{i}" for i in range(19)]
# Velocità angolare di rotazione per ciascuna delle 4 ruote
COLONNE_RUOTE = [f"wheelSpin_{i}" for i in range(4)]

# Vettore completo delle feature in ingresso alla rete neurale
COLONNE_INPUT = (
    COLONNE_TRACCIATO +
    ["speedX", "speedY", "speedZ", "trackPos", "angle", "rpm"] +
    COLONNE_RUOTE
)

# Costante utilizzata per mappare la marcia indietro (retromarcia) -1 a un valore positivo 0 per la CrossEntropy
OFFSET_MARCE = 1


# ──────────────────────────────────────────────────────────────────────────
# 2. CARICAMENTO E PRE-ELABORAZIONE DEI DATI (DATA PREPROCESSING)
# ──────────────────────────────────────────────────────────────────────────
def load_and_clean(percorso_csv: str) -> pd.DataFrame:
    """
    Carica il dataset da file CSV ed effettua la pulizia delle anomalie e il filtraggio.
    Filtra le traiettorie per considerare solo la guida pulita all'interno della pista.
    """
    dataframe_dati = pd.read_csv(percorso_csv)
    print(f"[PREPROCESSING] Numero iniziale di righe grezze: {len(dataframe_dati)}")
    
    # Rimozione delle righe che contengono valori mancanti (NaN) nelle colonne essenziali
    dataframe_dati.dropna(subset=COLONNE_INPUT + ['steer', 'accel', 'brake', 'gear'], inplace=True)
    
    # Se presente l'informazione sul giro, consideriamo solo i giri da 1 a 9 (evitando sessioni sporche)
    if 'lap' in dataframe_dati.columns:
        giri_validi = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataframe_dati = dataframe_dati[dataframe_dati['lap'].isin(giri_validi)]
        
    # Filtriamo tenendo solo i campioni in cui l'auto era ben allineata in pista (|trackPos| < 0.9)
    dataframe_dati = dataframe_dati[dataframe_dati['trackPos'].abs() < 0.9]
    print(f"[PREPROCESSING] Numero righe rimanenti dopo la pulizia: {len(dataframe_dati)}")
    
    return dataframe_dati


# ──────────────────────────────────────────────────────────────────────────
# 3. ARCHITETTURA RETE NEURALE (MLP MULTI-TASK)
# ──────────────────────────────────────────────────────────────────────────
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
        predizione_sterzo = torch.tanh(self.head_steer(strato_nascosto))
        predizione_pedali = torch.sigmoid(self.head_accel_brake(strato_nascosto))
        logits_marcia     = self.head_gear(strato_nascosto)
        return predizione_sterzo, predizione_pedali, logits_marcia


# ──────────────────────────────────────────────────────────────────────────
# 4. CICLO DI ADDESTRAMENTO PRINCIPALE (TRAINING PIPELINE)
# ──────────────────────────────────────────────────────────────────────────
def main():
    # Verifica correttezza degli argomenti da riga di comando
    if len(sys.argv) < 2:
        print("[ERRORE] Uso corretto del comando: python train_model.py <tuo_file_log.csv>")
        sys.exit(1)
        
    percorso_csv = sys.argv[1]
    dataframe_dati = load_and_clean(percorso_csv)

    # Estrazione delle matrici delle caratteristiche (X) e dei target (y)
    matrice_caratteristiche = dataframe_dati[COLONNE_INPUT].values.astype(np.float32)
    target_sterzo = dataframe_dati[['steer']].values.astype(np.float32)
    target_pedali = dataframe_dati[['accel', 'brake']].values.astype(np.float32)
    # Aggiungiamo un offset alle marce per evitare indici negativi nella CrossEntropy (la retromarcia diventa 0)
    target_marcia = (dataframe_dati['gear'].values + OFFSET_MARCE).astype(np.int64)

    # Calcolo dei parametri di normalizzazione (Media e Deviazione Standard) per la standardizzazione z-score
    media_caratteristiche = matrice_caratteristiche.mean(axis=0)
    deviazione_standard_caratteristiche = matrice_caratteristiche.std(axis=0) + 1e-6
    caratteristiche_normalizzate = (matrice_caratteristiche - media_caratteristiche) / deviazione_standard_caratteristiche

    # Salvataggio dei parametri di normalizzazione per l'inferenza real-time su TORCS
    informazioni_scaler = {
        "mean": media_caratteristiche,
        "std": deviazione_standard_caratteristiche,
        "input_cols": COLONNE_INPUT,
        "gear_offset": OFFSET_MARCE
    }
    joblib.dump(informazioni_scaler, "driving_scaler.pkl")
    print("[INFO] Scaler di normalizzazione salvato come 'driving_scaler.pkl'")

    # Creazione del dataset e suddivisione in Train Set (80%) e Validation Set (20%)
    dataset_completo = TensorDataset(
        torch.from_numpy(caratteristiche_normalizzate), 
        torch.from_numpy(target_sterzo), 
        torch.from_numpy(target_pedali), 
        torch.from_numpy(target_marcia)
    )
    dimensione_addestramento = int(0.8 * len(dataset_completo))
    dimensione_validazione = len(dataset_completo) - dimensione_addestramento
    dataset_addestramento, dataset_validazione = torch.utils.data.random_split(
        dataset_completo, [dimensione_addestramento, dimensione_validazione]
    )

    # Configurazione dei DataLoader per la gestione dei batch
    loader_addestramento = DataLoader(dataset_addestramento, batch_size=256, shuffle=True)
    loader_validazione = DataLoader(dataset_validazione, batch_size=512, shuffle=False)

    # Dispositivo hardware (CPU per l'addestramento su questo dataset di dimensioni contenute)
    dispositivo = torch.device("cpu")
    modello_guida = DrivingNet(dim_ingresso=len(COLONNE_INPUT)).to(dispositivo)
    
    # Definizione dell'ottimizzatore e del learning rate scheduler
    ottimizzatore = torch.optim.Adam(modello_guida.parameters(), lr=1e-3)
    schedulatore_lr = torch.optim.lr_scheduler.StepLR(ottimizzatore, step_size=20, gamma=0.5)
    
    # Criteri di Loss (Mean Squared Error per i compiti di regressione e Cross Entropy per la marcia)
    funzione_loss_mse = nn.MSELoss()
    funzione_loss_ce = nn.CrossEntropyLoss()

    miglior_loss_validazione = float("inf")
    epoche_totali = 60

    print("[INFO] Inizio del ciclo di addestramento...")
    for epoca in range(1, epoche_totali + 1):
        # Fase di training
        modello_guida.train()
        perdita_totale_epoca = 0.0
        for batch_x, batch_y_sterzo, batch_y_pedali, batch_y_marce in loader_addestramento:
            batch_x = batch_x.to(dispositivo)
            batch_y_sterzo = batch_y_sterzo.to(dispositivo)
            batch_y_pedali = batch_y_pedali.to(dispositivo)
            batch_y_marce = batch_y_marce.to(dispositivo)
            
            # Forward pass
            pred_sterzo, pred_pedali, pred_marce = modello_guida(batch_x)
            
            # Loss Multi-Task pesata per bilanciare i diversi obiettivi
            valore_loss = 2.0 * funzione_loss_mse(pred_sterzo, batch_y_sterzo) + \
                          1.0 * funzione_loss_mse(pred_pedali, batch_y_pedali) + \
                          0.3 * funzione_loss_ce(pred_marce, batch_y_marce)
            
            # Backward pass e ottimizzazione
            ottimizzatore.zero_grad()
            valore_loss.backward()
            ottimizzatore.step()
            
            perdita_totale_epoca += valore_loss.item() * batch_x.size(0)
            
        loss_addestramento = perdita_totale_epoca / len(dataset_addestramento)

        # Fase di validazione
        modello_guida.eval()
        perdita_validazione_cumulata = 0.0
        with torch.no_grad():
            for batch_x, batch_y_sterzo, batch_y_pedali, batch_y_marce in loader_validazione:
                batch_x = batch_x.to(dispositivo)
                batch_y_sterzo = batch_y_sterzo.to(dispositivo)
                batch_y_pedali = batch_y_pedali.to(dispositivo)
                batch_y_marce = batch_y_marce.to(dispositivo)
                
                pred_sterzo, pred_pedali, pred_marce = modello_guida(batch_x)
                valore_loss = 2.0 * funzione_loss_mse(pred_sterzo, batch_y_sterzo) + \
                              1.0 * funzione_loss_mse(pred_pedali, batch_y_pedali) + \
                              0.3 * funzione_loss_ce(pred_marce, batch_y_marce)
                
                perdita_validazione_cumulata += valore_loss.item() * batch_x.size(0)
                
        loss_validazione = perdita_validazione_cumulata / len(dataset_validazione)
        schedulatore_lr.step()

        segnalatore_salvataggio = ""
        # Verifica se abbiamo ottenuto una loss di validazione inferiore alla migliore registrata fino ad ora
        if loss_validazione < miglior_loss_validazione:
            miglior_loss_validazione = loss_validazione
            # Salvataggio (checkpoint) del modello migliore (Early Stopping empirico)
            torch.save(modello_guida.state_dict(), "driving_model.pt")
            segnalatore_salvataggio = " <- [Miglior Loss: Modello Salvato]"
            
        print(f"Epoca {epoca:3d}/{epoche_totali} | Loss Addestramento={loss_addestramento:.4f} | Loss Validazione={loss_validazione:.4f}{segnalatore_salvataggio}")

    print("\n[OK] Addestramento concluso con successo.")
    print("[INFO] Pesi del modello migliori salvati in: 'driving_model.pt'")


if __name__ == "__main__":
    main()