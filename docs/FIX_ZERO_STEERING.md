# Fix: Zero-Steering Root Cause — Implementation Plan

## Verified Findings (from actual model inspection)

### BC v2 Checkpoint
```
Input Dim: 8 features (no damage)
Architecture: 256 → 256 → 128 layers + LayerNorm + ReLU
Sensor Mean: [88.3, -0.07, 0.008, 6952, 3.97, 48.5, 43.3, 8.35]
Sensor Std:  [38.3, 0.30, 0.05, 1590, 1.45, 37.0, 31.1, 5.26]
```

**Features:** [speed, trackPos, angle, rpm, gear, track[6], track[12], track[18]]
**Scaling:** Z-score normalization using dataset statistics

### RL Models (both bc_warmstart and improved_reward_v1)
```
Input Dim: 9 features (includes damage)
Architecture: 256 → 256 (no hidden layer 3, no LayerNorm)
Activation: Tanh (vs BC's ReLU + LayerNorm)
```

**Features:** [speed, trackPos, angle, rpm, gear, damage, track[7], track[9], track[11]]
**Scaling:** Hardcoded divisors (speed/300, rpm/10000, etc.) — NO z-score normalization

---

## Root Causes (Ranked by Impact)

### 🔴 CRITICAL #1: Input Normalization Mismatch

**Problem:**
- BC trained on z-score normalized inputs learned from telemetry statistics
- RL trained on raw scaled inputs with hardcoded divisors
- These are fundamentally different input distributions

**Evidence:**
```
BC speed normalization:    (speed - 88.3) / 38.3
RL speed normalization:    speed / 300.0

At speed 150 km/h:
  BC:  (150 - 88.3) / 38.3 = +1.61  (positive, above mean)
  RL:  150 / 300 = 0.5                (middle of scale, looks "normal")
```

**Impact on steering:**
- RL sees high speeds as "normal" (0.5 scale)
- Steering angle is also 0–1 scale
- Model learns: both are equally important
- Optimization plateau: steering signal gets suppressed

### 🔴 CRITICAL #2: Track Sensor Index Mismatch

**Problem:**
```
BC uses track indices:   [6, 12, 18]  (approximately -9°, 0°, +81°?)
RL uses track indices:   [7, 9, 11]   (approximately -6°, 0°, +6°)
```

**Impact:**
- RL has much narrower forward lookahead (±6° vs ±9°+)
- Corkscrew has sharp corners requiring wider vision
- Steering decision made too late in the corner

### 🟠 HIGH #3: Network Architecture Divergence

**Problem:**
```
BC:  [in=8] → ReLU + LN → [256] → ReLU + LN → [256] → ReLU + LN → [128]
RL:  [in=9] → Tanh → [256] → Tanh → [256]
```

**Impact:**
- LayerNorm stabilizes gradients; RL doesn't have it
- ReLU + LayerNorm is more numerically stable than Tanh
- Extra hidden layer [128] provides more capacity
- RL's shallower network may not learn steering well

### 🟡 MEDIUM #4: Damage Feature Noise

**Problem:**
- Damage is sparse (mostly 0 until crash)
- Adds dimensionality without information
- May dilute steering signal during optimization

---

## Fix Strategy (3-Phase Approach)

### Phase 1: Align Observation Space

**Change gym_env.py `_make_obs()` to match BC:**

```python
def _make_obs(self, state: SensorState) -> np.ndarray:
    raw = np.array([
        state.speed,
        state.trackPos,
        state.angle,
        state.rpm,
        float(self._gear),
        # REMOVED: state.damage
        state.track[6],   # CHANGED from 7
        state.track[12],  # CHANGED from 9
        state.track[18],  # CHANGED from 11
    ], dtype=np.float32)
    
    # Z-score normalize using BC statistics
    # (Load from BC checkpoint or compute from random rollouts)
    normalized = (raw - self._sensor_mean) / self._sensor_std
    return normalized
```

**Files to modify:**
- `training/rl/gym_env.py`: Update `_make_obs()` and add normalization stats
- `drivers/rl/driver.py`: Update `_make_obs()` to match

**Why this works:**
- RL now sees same input distribution as BC training data
- BC warm-start initialization becomes meaningful
- Track sensors aligned with BC's lookahead geometry

---

### Phase 2: Load BC Normalization Stats

**Option A: From BC checkpoint (recommended)**

```python
# In gym_env.py __init__:
import torch
from pathlib import Path

bc_path = Path(__file__).resolve().parents[2] / "models" / "bc_v2.pth"
try:
    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    self._sensor_mean = torch.from_numpy(ckpt["sensor_mean"]).float()
    self._sensor_std = torch.from_numpy(ckpt["sensor_std"]).float()
except Exception as e:
    logger.warning(f"Could not load BC normalization stats: {e}")
    # Fallback to hardcoded empirical values from bc_v2.pth
    self._sensor_mean = torch.tensor([88.3, -0.07, 0.008, 6952, 3.97, 48.5, 43.3, 8.35])
    self._sensor_std = torch.tensor([38.3, 0.30, 0.05, 1590, 1.45, 37.0, 31.1, 5.26])
```

**Option B: Compute from random rollouts**

- Slower but doesn't require BC checkpoint
- Recommended if BC is unavailable

---

### Phase 3: Enhance Network Architecture (Optional but Recommended)

**Update train_rl_bc_warmstart.py:**

```python
policy_kwargs=dict(
    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # Add [128] layer
    activation_fn=nn.ReLU,  # Use ReLU instead of Tanh
)
```

**Benefits:**
- Matches BC architecture more closely
- ReLU + LayerNorm more stable for RL
- Extra capacity for fine-tuning

---

## Implementation Steps

### Step 1: Modify gym_env.py

```python
# In TORCSGymEnv.__init__():
self._sensor_mean = None
self._sensor_std = None
self._load_normalization_stats()

def _load_normalization_stats(self) -> None:
    """Load BC normalization stats to match input distribution."""
    try:
        import torch
        from pathlib import Path
        bc_path = Path(__file__).resolve().parents[2] / "models" / "bc_v2.pth"
        ckpt = torch.load(str(bc_path), map_location="cpu", weights_only=False)
        # Use first 8 features of RL's 9-feature obs (skip damage)
        mean = ckpt["sensor_mean"]
        std = ckpt["sensor_std"]
    except Exception as e:
        # Fallback: hardcoded empirical stats from bc_v2
        import numpy as np
        mean = np.array([88.3, -0.07, 0.008, 6952, 3.97, 48.5, 43.3, 8.35], dtype=np.float32)
        std = np.array([38.3, 0.30, 0.05, 1590, 1.45, 37.0, 31.1, 5.26], dtype=np.float32)
        logger.warning(f"Using fallback BC stats: {e}")
    
    self._sensor_mean = mean
    self._sensor_std = std

def _make_obs(self, state: SensorState) -> np.ndarray:
    raw = np.array([
        state.speed,
        state.trackPos,
        state.angle,
        state.rpm,
        float(self._gear),
        state.track[6],
        state.track[12],
        state.track[18],
    ], dtype=np.float32)
    
    normalized = (raw - self._sensor_mean) / (self._sensor_std + 1e-8)
    return normalized
```

### Step 2: Update RL Driver

```python
# In drivers/rl/driver.py, update _make_obs() to match gym_env
def _make_obs(self, state: SensorState) -> np.ndarray:
    # Must match gym_env.py exactly
    raw = np.array([
        state.speed,
        state.trackPos,
        state.angle,
        state.rpm,
        float(self._gear),
        state.track[6],
        state.track[12],
        state.track[18],
    ], dtype=np.float32)
    
    # Use same normalization as training
    # Fallback to hardcoded BC stats if needed
    mean = np.array([88.3, -0.07, 0.008, 6952, 3.97, 48.5, 43.3, 8.35], dtype=np.float32)
    std = np.array([38.3, 0.30, 0.05, 1590, 1.45, 37.0, 31.1, 5.26], dtype=np.float32)
    
    return (raw - mean) / (std + 1e-8)
```

### Step 3: Retrain RL

```bash
# After applying changes, retrain with BC warm-start:
conda run -n ai_env python training/rl/train_rl_bc_warmstart.py \
    --bc-model models/bc_v2.pth \
    --target-steps 100000 \
    --sessions 60 \
    --save-path models/rl_bc_warmstart_v3_fixed
```

### Step 4: Verify Steering Output

```bash
# Test the new model on one lap
conda run -n ai_env python scripts/run_agent.py \
    --driver rl_rl_bc_warmstart_v3_fixed \
    --laps 1 \
    --telemetry
```

**Check telemetry CSV for:**
- Non-zero steering values during curves (not zeros)
- Progressive steering (not on/off binary)
- Lap completion (not crash at ~3.2 km)

---

## Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Steering output | Zero on curves | Non-zero, progressive |
| Observation match | RL ≠ BC | RL = BC inputs |
| Lap completion | ~3.2 km crash | Full lap, < 150 s |
| Track lookahead | ±6° (narrow) | ±9° (wider, matches BC) |

---

## Rollback Plan

If changes break the model:
1. Revert gym_env.py and rl driver.py to commit `56547e5`
2. The `rl_bc_warmstart/final.zip` checkpoint remains valid
3. Use `--driver rl_bc_warmstart` as baseline

---

## Next Steps After Fix

Once steering is working:
1. Monitor reward function again (may need fine-tuning now that model can steer)
2. Consider wider track sensors [4, 9, 14] for sharper curves
3. Explore larger networks [512, 512, 256] for RL fine-tuning
4. Measure lap time improvement vs. rule-based (target: < 148 s)
