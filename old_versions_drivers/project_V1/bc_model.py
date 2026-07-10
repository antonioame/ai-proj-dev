"""
Modello di Behavioral Cloning per le corse in TORCS
Impara dalle azioni manuali per controllare il veicolo
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

class BehavioralCloningModel:
    def __init__(self, model_type='neural'):
        self.model_type = model_type
        
        # Scalers per input e output
        self.input_scaler = StandardScaler()
        self.output_scalers = {
            'steer': StandardScaler(),
            'accel': StandardScaler(),
            'brake': StandardScaler(),
            'gear': StandardScaler()
        }
        
        # Modelli per ogni azione
        if model_type == 'neural':
            self.models = {
                'steer': MLPRegressor(hidden_layer_sizes=(64, 32), 
                                     activation='relu', learning_rate_init=0.005,
                                     max_iter=200, batch_size=512, early_stopping=True, random_state=42),
                'accel': MLPRegressor(hidden_layer_sizes=(64, 32), 
                                     activation='relu', learning_rate_init=0.005,
                                     max_iter=200, batch_size=512, early_stopping=True, random_state=42),
                'brake': MLPRegressor(hidden_layer_sizes=(64, 32), 
                                     activation='relu', learning_rate_init=0.005,
                                     max_iter=200, batch_size=512, early_stopping=True, random_state=42),
                'gear': MLPRegressor(hidden_layer_sizes=(32, 16), 
                                    activation='relu', learning_rate_init=0.005,
                                    max_iter=200, batch_size=512, early_stopping=True, random_state=42)
            }
        else:  # Random Forest
            self.models = {
                'steer': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
                'accel': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
                'brake': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
                'gear': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            }
        
        self.is_trained = False
    
    def load_expert_data(self, csv_paths):
        """Carica i dati aggregati dai vari file CSV dell'esperto"""
        if isinstance(csv_paths, str):
            if os.path.isdir(csv_paths):
                # Se è una directory, prendi tutti i .csv
                csv_paths = [os.path.join(csv_paths, f) for f in os.listdir(csv_paths) if f.endswith('.csv')]
            else:
                csv_paths = [csv_paths]
                
        print(f"Caricamento dati da {len(csv_paths)} file(s)...")
        
        dataframes = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                dataframes.append(df)
                print(f"  - {path}: {len(df)} campioni")
            except Exception as e:
                print(f"  - Errore in {path}: {e}")
                
        if not dataframes:
            raise ValueError("Nessun dato valido trovato!")
            
        # Concatena tutti i dataset
        df_combined = pd.concat(dataframes, ignore_index=True)
        
        # Filtrare i dati validi
        df_combined = df_combined.dropna()
        
        # Feature di input (sensori)
        self.sensor_columns = ['speedX', 'trackPos', 'angle', 'rpm', 'damage']
        
        # Selezionare solo i sensori rilevanti come input
        X = df_combined[['speedX', 'trackPos', 'angle', 'rpm', 'damage']].values
        
        # Output: azioni dell'esperto
        y_steer = df_combined['steer'].values.reshape(-1, 1)
        y_accel = df_combined['accel'].values.reshape(-1, 1)
        y_brake = df_combined['brake'].values.reshape(-1, 1)
        y_gear = df_combined['gear'].values.reshape(-1, 1)
        
        print(f"\nDataset totale caricato: {len(df_combined)} campioni")
        print(f"Intervallo steer: [{y_steer.min():.3f}, {y_steer.max():.3f}]")
        print(f"Intervallo accel: [{y_accel.min():.3f}, {y_accel.max():.3f}]")
        print(f"Intervallo brake: [{y_brake.min():.3f}, {y_brake.max():.3f}]")
        print(f"Intervallo gear: [{y_gear.min():.0f}, {y_gear.max():.0f}]")
        
        return X, y_steer, y_accel, y_brake, y_gear
    
    def train(self, csv_paths, test_size=0.2):
        """Allena il modello aggregato sui dati manuali"""
        X, y_steer, y_accel, y_brake, y_gear = self.load_expert_data(csv_paths)
        
        # Normalizzare input
        print("\nNormalizzazione dati...")
        X_scaled = self.input_scaler.fit_transform(X)
        
        # Normalizzare output
        y_steer_scaled = self.output_scalers['steer'].fit_transform(y_steer)
        y_accel_scaled = self.output_scalers['accel'].fit_transform(y_accel)
        y_brake_scaled = self.output_scalers['brake'].fit_transform(y_brake)
        y_gear_scaled = self.output_scalers['gear'].fit_transform(y_gear)
        
        # Addestrare i modelli
        print(f"Addestramento modelli ({self.model_type})...")
        
        print("  - Steering...")
        self.models['steer'].fit(X_scaled, y_steer_scaled.ravel())
        steer_train_score = self.models['steer'].score(X_scaled, y_steer_scaled.ravel())
        print(f"    Score: {steer_train_score:.4f}")
        
        print("  - Accelerazione...")
        self.models['accel'].fit(X_scaled, y_accel_scaled.ravel())
        accel_train_score = self.models['accel'].score(X_scaled, y_accel_scaled.ravel())
        print(f"    Score: {accel_train_score:.4f}")
        
        print("  - Freno...")
        self.models['brake'].fit(X_scaled, y_brake_scaled.ravel())
        brake_train_score = self.models['brake'].score(X_scaled, y_brake_scaled.ravel())
        print(f"    Score: {brake_train_score:.4f}")
        
        print("  - Cambio...")
        self.models['gear'].fit(X_scaled, y_gear_scaled.ravel())
        gear_train_score = self.models['gear'].score(X_scaled, y_gear_scaled.ravel())
        print(f"    Score: {gear_train_score:.4f}")
        
        self.is_trained = True
        print("\nAddestramento completato!")
    
    def predict_actions(self, state_dict):
        """Predice le azioni basate sullo stato corrente"""
        if not self.is_trained:
            raise ValueError("Modello non addestrato! Esegui train() prima.")
        
        # Estrarre i sensori dal dizionario di stato
        state_values = np.array([[
            state_dict.get('speedX', 0),
            state_dict.get('trackPos', 0),
            state_dict.get('angle', 0),
            state_dict.get('rpm', 0),
            state_dict.get('damage', 0)
        ]])
        
        # Normalizzare
        state_scaled = self.input_scaler.transform(state_values)
        
        # Predire (output è normalizzato, bisogna denormalizzare)
        steer_scaled = self.models['steer'].predict(state_scaled)[0].reshape(1, -1)
        accel_scaled = self.models['accel'].predict(state_scaled)[0].reshape(1, -1)
        brake_scaled = self.models['brake'].predict(state_scaled)[0].reshape(1, -1)
        gear_scaled = self.models['gear'].predict(state_scaled)[0].reshape(1, -1)
        
        # Denormalizzare
        steer = self.output_scalers['steer'].inverse_transform(steer_scaled)[0, 0]
        accel = self.output_scalers['accel'].inverse_transform(accel_scaled)[0, 0]
        brake = self.output_scalers['brake'].inverse_transform(brake_scaled)[0, 0]
        gear = round(float(self.output_scalers['gear'].inverse_transform(gear_scaled)[0, 0]))
        
        return {
            'steer': float(np.clip(steer, -1.0, 1.0)),
            'accel': float(np.clip(accel, 0.0, 1.0)),
            'brake': float(np.clip(brake, 0.0, 1.0)),
            'gear': int(np.clip(gear, 1, 5))
        }
    
    def save(self, model_dir='bc_models'):
        """Salva il modello addestrato"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Salvare gli scaler e i modelli
        with open(os.path.join(model_dir, 'input_scaler.pkl'), 'wb') as f:
            pickle.dump(self.input_scaler, f)
        
        for action in ['steer', 'accel', 'brake', 'gear']:
            with open(os.path.join(model_dir, f'{action}_scaler.pkl'), 'wb') as f:
                pickle.dump(self.output_scalers[action], f)
            with open(os.path.join(model_dir, f'{action}_model.pkl'), 'wb') as f:
                pickle.dump(self.models[action], f)
        
        print(f"\nModello salvato in {model_dir}/")
    
    def load(self, model_dir='bc_models'):
        """Carica un modello addestrato precedentemente"""
        with open(os.path.join(model_dir, 'input_scaler.pkl'), 'rb') as f:
            self.input_scaler = pickle.load(f)
        
        for action in ['steer', 'accel', 'brake', 'gear']:
            with open(os.path.join(model_dir, f'{action}_scaler.pkl'), 'rb') as f:
                self.output_scalers[action] = pickle.load(f)
            with open(os.path.join(model_dir, f'{action}_model.pkl'), 'rb') as f:
                self.models[action] = pickle.load(f)
        
        self.is_trained = True
        print(f"Modello caricato da {model_dir}/")


if __name__ == "__main__":
    # Addestrare il modello su tutta la cartella manual_logs (include sia session-001.csv che session-002.csv)
    bc = BehavioralCloningModel(model_type='neural')
    bc.train('manual_logs')
    bc.save('bc_models')
    
    # Test: Predicere azioni
    test_state = {
        'speedX': 50.0,
        'trackPos': 0.0,
        'angle': 0.0,
        'rpm': 5000,
        'damage': 0
    }
    
    actions = bc.predict_actions(test_state)
    print(f"\nTest prediction per stato: {test_state}")
    print(f"Azioni predette: {actions}")
