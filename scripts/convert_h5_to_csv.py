"""Convert Bartolo1024 HDF5 dataset to CSV format compatible with our BC model."""

import sys
from pathlib import Path
import h5py
import numpy as np
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def convert_h5_to_csv(h5_path, csv_output):
    """Convert HDF5 dataset to CSV.

    Bartolo format inspection via train_input.py:
    - Load 'sa' array from HDF5
    - Delete column 2 (gear)
    - If merged: accel_merged = accel - brake, delete brake col

    This means original format before deletions:
    [steer, accel, gear, brake, sensor_0-28]  (34 cols)

    After first delete (gear): [steer, accel, brake, sensor_0-28]  (33 cols)
    After merge: [steer, accel_merged, sensor_0-28]  (32 cols)

    We keep ALL 4 separate actions for better control.
    """
    print(f"Loading HDF5 from {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        data = np.array(f.get('sa'))

    print(f"Loaded {data.shape[0]} samples")
    print(f"Data shape: {data.shape}")
    print(f"Columns: 0=steer, 1=accel, 2=gear, 3=brake, 4-32=sensors(29)")

    # Write CSV
    with open(csv_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header (matching our dataset.py expectations)
        header = [
            'angle', 'speed', 'speedY', 'speedZ', 'trackPos',
            'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
            'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
            'track_10', 'track_11', 'track_12', 'track_13', 'track_14',
            'track_15', 'track_16', 'track_17', 'track_18',
            'rpm', 'gear',  # sensors (26 total)
            'steer', 'accel', 'brake', 'gear_cmd'  # actions (4)
        ]
        writer.writerow(header)

        # Write data rows
        # Original: [steer, accel, gear, brake, sensor_0-28]
        for i, row in enumerate(data):
            steer = float(row[0])
            accel = float(row[1])
            # gear = float(row[2])  # Skip gear from input; it's part of sensor
            brake = float(row[3])
            sensors = row[4:]  # 29 sensors (indices 4-32)

            # Bartolo's 29 sensors → our 26 (take first 26)
            sensor_data = sensors[:26].tolist() if len(sensors) >= 26 else list(sensors[:26]) + [0] * (26 - len(sensors))

            # Gear is not an action output for Bartolo, but part of state
            # So we just use a fixed gear for the action column
            gear_cmd = 3

            row_data = sensor_data + [steer, accel, brake, gear_cmd]
            writer.writerow(row_data)

            if (i + 1) % 10000 == 0:
                print(f"  Wrote {i + 1} rows...")

    print(f"✓ Converted to CSV: {csv_output}")
    print(f"  Total samples: {data.shape[0]}")
    return csv_output


if __name__ == "__main__":
    h5_file = Path(__file__).resolve().parent.parent / "downloaded" / "TORCS-behavioral-cloning-master" / "train_data" / "alldata.h5"
    csv_file = Path(__file__).resolve().parent.parent / "data" / "bartolo_converted.csv"

    if not h5_file.exists():
        print(f"Error: {h5_file} not found")
        sys.exit(1)

    csv_file.parent.mkdir(exist_ok=True)
    convert_h5_to_csv(str(h5_file), str(csv_file))
