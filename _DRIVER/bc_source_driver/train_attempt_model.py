"""
Train the earlier driving-net attempt using existing telemetry CSV.
Usage:
    conda run -n ai_env python _DRIVER/bc_source_driver/train_attempt_model.py --csv data/rule_based_20260628_203648.csv
"""

import sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse

# Input feature columns (matches actual CSV format from our telemetry)
COLONNE_TRACCIATO = [f"track_{i}" for i in range(19)]
COLONNE_RUOTE = [f"wheel_{i}" for i in range(4)]
COLONNE_INPUT = (
    COLONNE_TRACCIATO +
    ["speed", "trackPos", "angle", "rpm"] +
    COLONNE_RUOTE
)
OFFSET_MARCE = 1


class DrivingNet(nn.Module):
    """Multi-Layer Perceptron for driving control."""
    def __init__(self, dim_ingresso: int, numero_marce: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim_ingresso, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head_steer = nn.Linear(64, 1)
        self.head_accel_brake = nn.Linear(64, 2)
        self.head_gear = nn.Linear(64, numero_marce)

    def forward(self, dati_ingresso):
        strato_nascosto = self.backbone(dati_ingresso)
        return (
            torch.tanh(self.head_steer(strato_nascosto)),
            torch.sigmoid(self.head_accel_brake(strato_nascosto)),
            self.head_gear(strato_nascosto),
        )


def load_and_clean(percorso_csv: str) -> pd.DataFrame:
    """Load and clean telemetry CSV."""
    dataframe_dati = pd.read_csv(percorso_csv)
    print(f"[PREPROCESSING] Raw rows: {len(dataframe_dati)}")

    # Drop rows with missing values in critical columns
    dataframe_dati.dropna(subset=COLONNE_INPUT + ['steer', 'accel', 'brake', 'gear'], inplace=True)

    # Filter to clean driving (on track)
    dataframe_dati = dataframe_dati[dataframe_dati['trackPos'].abs() < 0.9]
    print(f"[PREPROCESSING] Rows after cleaning: {len(dataframe_dati)}")

    return dataframe_dati


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to telemetry CSV")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "attempt_model"),
                        help="Output directory for model")
    args = parser.parse_args()

    # Load and preprocess
    dataframe_dati = load_and_clean(args.csv)
    matrice_caratteristiche = dataframe_dati[COLONNE_INPUT].values.astype(np.float32)
    target_sterzo = dataframe_dati[['steer']].values.astype(np.float32)
    target_pedali = dataframe_dati[['accel', 'brake']].values.astype(np.float32)
    target_marcia = (dataframe_dati['gear'].values + OFFSET_MARCE).astype(np.int64)

    # Normalize
    media_caratteristiche = matrice_caratteristiche.mean(axis=0)
    deviazione_standard_caratteristiche = matrice_caratteristiche.std(axis=0) + 1e-6
    caratteristiche_normalizzate = (matrice_caratteristiche - media_caratteristiche) / deviazione_standard_caratteristiche

    # Save scaler
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    informazioni_scaler = {
        "mean": media_caratteristiche,
        "std": deviazione_standard_caratteristiche,
        "input_cols": COLONNE_INPUT,
        "gear_offset": OFFSET_MARCE
    }
    scaler_path = output_path / "driving_scaler.pkl"
    joblib.dump(informazioni_scaler, scaler_path)
    print(f"[INFO] Scaler saved: {scaler_path}")

    # Create dataset
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

    loader_addestramento = DataLoader(dataset_addestramento, batch_size=256, shuffle=True)
    loader_validazione = DataLoader(dataset_validazione, batch_size=512, shuffle=False)

    # Train
    dispositivo = torch.device("cpu")
    modello_guida = DrivingNet(dim_ingresso=len(COLONNE_INPUT)).to(dispositivo)
    ottimizzatore = torch.optim.Adam(modello_guida.parameters(), lr=1e-3)
    schedulatore_lr = torch.optim.lr_scheduler.StepLR(ottimizzatore, step_size=20, gamma=0.5)
    funzione_loss_mse = nn.MSELoss()
    funzione_loss_ce = nn.CrossEntropyLoss()

    miglior_loss_validazione = float("inf")
    epoche_totali = 60

    print("[INFO] Training started...")
    for epoca in range(1, epoche_totali + 1):
        modello_guida.train()
        perdita_totale_epoca = 0.0
        for batch_x, batch_y_sterzo, batch_y_pedali, batch_y_marce in loader_addestramento:
            batch_x, batch_y_sterzo, batch_y_pedali, batch_y_marce = (
                batch_x.to(dispositivo), batch_y_sterzo.to(dispositivo),
                batch_y_pedali.to(dispositivo), batch_y_marce.to(dispositivo)
            )
            pred_sterzo, pred_pedali, pred_marce = modello_guida(batch_x)
            valore_loss = (
                2.0 * funzione_loss_mse(pred_sterzo, batch_y_sterzo) +
                1.0 * funzione_loss_mse(pred_pedali, batch_y_pedali) +
                0.3 * funzione_loss_ce(pred_marce, batch_y_marce)
            )
            ottimizzatore.zero_grad()
            valore_loss.backward()
            ottimizzatore.step()
            perdita_totale_epoca += valore_loss.item() * batch_x.size(0)

        loss_addestramento = perdita_totale_epoca / len(dataset_addestramento)

        modello_guida.eval()
        perdita_validazione_cumulata = 0.0
        with torch.no_grad():
            for batch_x, batch_y_sterzo, batch_y_pedali, batch_y_marce in loader_validazione:
                batch_x, batch_y_sterzo, batch_y_pedali, batch_y_marce = (
                    batch_x.to(dispositivo), batch_y_sterzo.to(dispositivo),
                    batch_y_pedali.to(dispositivo), batch_y_marce.to(dispositivo)
                )
                pred_sterzo, pred_pedali, pred_marce = modello_guida(batch_x)
                valore_loss = (
                    2.0 * funzione_loss_mse(pred_sterzo, batch_y_sterzo) +
                    1.0 * funzione_loss_mse(pred_pedali, batch_y_pedali) +
                    0.3 * funzione_loss_ce(pred_marce, batch_y_marce)
                )
                perdita_validazione_cumulata += valore_loss.item() * batch_x.size(0)

        loss_validazione = perdita_validazione_cumulata / len(dataset_validazione)
        schedulatore_lr.step()

        segnalatore_salvataggio = ""
        if loss_validazione < miglior_loss_validazione:
            miglior_loss_validazione = loss_validazione
            model_path = output_path / "driving_model.pt"
            torch.save(modello_guida.state_dict(), model_path)
            segnalatore_salvataggio = " <- [SAVED]"

        print(f"Epoch {epoca:3d}/{epoche_totali} | Train Loss={loss_addestramento:.4f} | Val Loss={loss_validazione:.4f}{segnalatore_salvataggio}")

    print(f"\n[OK] Training complete. Model saved to {output_path / 'driving_model.pt'}")


if __name__ == "__main__":
    main()
