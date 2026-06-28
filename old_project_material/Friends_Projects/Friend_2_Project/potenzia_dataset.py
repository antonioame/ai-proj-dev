import json
import random
import os
import copy

def augment_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Errore: {input_file} non trovato.")
        return

    print(f"Caricamento {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Record originali: {len(data)}")

    # 1. MIRRORING TOTALE
    # Raddoppia i dati e garantisce simmetria (fondamentale per pochi dati)
    print("Applicazione Mirroring Totale...")
    mirrored_data = []
    for r in data:
        mirrored = copy.deepcopy(r)
        
        # Inverti i comandi e i sensori direzionali
        mirrored['action_steer'] = -r['action_steer']
        mirrored['angle'] = -r['angle']
        mirrored['trackPos'] = -r['trackPos']
        
        # Inverti i sensori di distanza (19 sensori)
        mirrored['track'] = r['track'][::-1]
        
        mirrored_data.append(mirrored)
    
    combined_data = data + mirrored_data
    print(f"Dataset dopo Mirroring: {len(combined_data)} record.")

    # 2. BILANCIAMENTO E POTENZIAMENTO AGGRESSIVO
    # Dividiamo il dataset in tre categorie
    straight_records = [r for r in combined_data if abs(r['action_steer']) < 0.05]
    curve_records = [r for r in combined_data if abs(r['action_steer']) >= 0.05]
    
    print(f"Record rettilinei (<0.05): {len(straight_records)}")
    print(f"Record curve (>=0.05): {len(curve_records)}")

    # Sottocampioniamo i rettilinei (ne teniamo meno ma non troppo per mantenere stabilità)
    random.shuffle(straight_records)
    straight_keep = straight_records[:int(len(straight_records) * 0.5)]
    
    augmented_data = straight_keep + curve_records
    print(f"Dataset dopo sottocampionamento bilanciato rettilinei (50%): {len(augmented_data)} record.")

    def add_noisy_records(source_list, factor, label, inject_brake=False):
        print(f"Generazione record extra per {label} (x{factor})...")
        for _ in range(factor):
            for r in source_list:
                new_record = copy.deepcopy(r)
                
                # Logica Braking Instructor DINAMICA
                if inject_brake and abs(r['action_steer']) > 0.1:
                    speed = r['speedX']
                    if speed > 80:
                        # Troppo veloce: frena e togli gas
                        new_record['action_brake'] = 0.3
                        new_record['action_accel'] = 0.0
                    elif speed > 40:
                        # Velocità media: parzializza il gas
                        new_record['action_brake'] = 0.0
                        new_record['action_accel'] = 0.4
                    else:
                        # Troppo lenta: dai gas
                        new_record['action_brake'] = 0.0
                        new_record['action_accel'] = 0.7
                
                # (Logica del rumore rimossa)
                
                augmented_data.append(new_record)

    # Potenziamo le curve iniettando la logica di frenata (x5)
    add_noisy_records(curve_records, 5, "Curve con Freno", inject_brake=True)
    
    # Potenziamo i rettilinei rimasti per dare stabilità (x1)
    add_noisy_records(straight_keep, 1, "Stabilità Rettilineo", inject_brake=False)

    # 3. SHUFFLE E SALVATAGGIO
    random.shuffle(augmented_data)
    
    print(f"Salvataggio in {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=4)
    
    print(f"Completato! Nuovo dataset: {len(augmented_data)} record totali.")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    input_f = os.path.join(script_dir, 'tutto.json')
    output_f = os.path.join(script_dir, 'tutto_potenziato.json')
    augment_dataset(input_f, output_f)
