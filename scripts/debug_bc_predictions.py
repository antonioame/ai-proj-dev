"""Debug BC model predictions on sample data."""

import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bc_driver.driver import BCDriver
from torcs_env.sensors import SensorState

# Load a sample from the CSV
csv_path = Path(__file__).resolve().parent.parent / "data" / "bartolo_converted.csv"
df = pd.read_csv(csv_path, nrows=100)

# Create a dummy sensor state from first row
row = df.iloc[0]
state = SensorState(
    angle=float(row['angle']),
    speed=float(row['speed']),
    speedY=float(row['speedY']),
    speedZ=float(row['speedZ']),
    trackPos=float(row['trackPos']),
    track=tuple(float(row[f'track_{i}']) for i in range(19)),
    rpm=float(row['rpm']),
    gear=int(row['gear']),
    distRaced=0.0,
    curLapTime=0.0,
    lastLapTime=0.0,
)

# Load BC driver
driver = BCDriver()

# Get prediction
action = driver.step(state)

print("Input state:")
print(f"  angle={state.angle:.4f}")
print(f"  speed={state.speed:.4f}")
print(f"  trackPos={state.trackPos:.4f}")
print(f"  track[0:5]={state.track[0:5]}")
print(f"  rpm={state.rpm:.1f}")
print(f"  gear={state.gear}")

print("\nCSV row (first 10 values):")
print(df.iloc[0].head(10).to_dict())

print("\nModel prediction:")
print(f"  steer={action.steer:.4f}")
print(f"  accel={action.accel:.4f}")
print(f"  brake={action.brake:.4f}")
print(f"  gear={action.gear}")

print("\nRaw action (before clamping):")
sensor_vec = np.array([
    state.angle,
    state.speed,
    state.speedY,
    state.speedZ,
    state.trackPos,
    *state.track,
    state.rpm,
    float(state.gear),
], dtype=np.float32)

sensor_tensor = torch.from_numpy(sensor_vec).float().to(driver._device)
sensor_tensor = (sensor_tensor - driver.X_mean) / driver.X_std

with torch.no_grad():
    action_pred = driver.model(sensor_tensor).cpu().numpy()

print(f"  Raw steer={action_pred[0]:.4f}")
print(f"  Raw accel={action_pred[1]:.4f}")
print(f"  Raw brake={action_pred[2]:.4f}")
print(f"  Raw gear={action_pred[3]:.4f}")

print("\nCSV action targets (first 10 rows):")
print(df[['steer', 'accel', 'brake', 'gear_cmd']].head(10).describe())
