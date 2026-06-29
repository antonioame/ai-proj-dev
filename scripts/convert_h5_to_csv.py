"""Convert Bartolo1024 HDF5 dataset to CSV format compatible with our BC model."""

import sys
from pathlib import Path
import h5py
import numpy as np
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def convert_h5_to_csv(h5_path, csv_output):
    """Convert HDF5 dataset to CSV.

    H5 format (from train_input.py):
    - Column 0: steering
    - Column 1: acceleration
    - Column 2: brake
    - Column 3+: 29 sensors

    CSV format (for our BC):
    - Sensors: angle, speed, speedY, speedZ, trackPos, track_0-18, rpm, gear (26 total)
    - Actions: steer, accel, brake, gear_cmd (4 total)
    """
    print(f"Loading HDF5 from {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        data = np.array(f.get('sa'))  # shape: (N, 32) — steering, accel, brake, + 29 sensors

    print(f"Loaded {data.shape[0]} samples")
    print(f"Data shape: {data.shape}")

    # Write CSV
    with open(csv_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header (matching our dataset.py expectations)
        # Note: Bartolo's data doesn't have all sensor types, so we'll map generically
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
        # Bartolo's data: [steering, accel, brake, sensor_0, ..., sensor_28]
        for i, row in enumerate(data):
            steer = row[0]
            accel = row[1]
            brake = row[2]
            sensors = row[3:]  # 29 sensors

            # Map sensors: assume they're track range-finders + some others
            # Bartolo's 29 sensors → our 26 (take first 26, skip last 3)
            sensor_data = sensors[:26] if len(sensors) >= 26 else list(sensors) + [0] * (26 - len(sensors))

            # Assume gear is constant (auto)
            gear_cmd = 3  # mid-range gear

            row_data = list(sensor_data) + [steer, accel, brake, gear_cmd]
            writer.writerow(row_data)

            if (i + 1) % 50000 == 0:
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
