"""Debug BC model to see what it's outputting."""
import torch
import numpy as np
from pathlib import Path
from training.behavioral_cloning.model import MLPPolicy

model_path = Path("models/bc_v1.pth")
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

print("Checkpoint keys:", ckpt.keys())
print("Input dim:", ckpt["input_dim"])
print("Output dim:", ckpt["output_dim"])
print("Hidden dims:", ckpt["hidden_dims"])
print("Sensor mean:", ckpt["sensor_mean"])
print("Sensor std:", ckpt["sensor_std"])

# Load model
model = MLPPolicy(
    input_dim=ckpt["input_dim"],
    hidden_dims=ckpt["hidden_dims"],
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Test with typical inputs from rule-based driver at start
# ["speed", "trackPos", "angle", "rpm", "gear"]
test_inputs = [
    [0.0, 0.33, 0.01, 7869, 0],     # Starting
    [39.3, 0.29, 0.05, 6218, 1],    # Driving straight
    [75.4, -0.63, 0.22, 11940, 1],  # Turning
    [88.1, -1.54, 0.22, 14118, 1],  # Big turn
]

mean = torch.from_numpy(ckpt["sensor_mean"].astype(np.float32))
std = torch.from_numpy(ckpt["sensor_std"].astype(np.float32))

for i, inputs in enumerate(test_inputs):
    x = torch.tensor(inputs, dtype=torch.float32)
    x_norm = (x - mean) / std
    print(f"\nTest {i}: raw={inputs}")
    print(f"  normalized={x_norm.tolist()}")

    x_batch = x_norm.unsqueeze(0)
    out = model.predict(x_batch)

    print(f"  Model output:")
    print(f"    steer: {out['steer'].item():.6f}")
    print(f"    accel: {out['accel'].item():.6f}")
    print(f"    brake: {out['brake'].item():.6f}")
    print(f"    gear: {out['gear'].item():.0f}")
