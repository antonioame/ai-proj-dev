# Investigation: Zero-Steering Root Cause

## Summary

The RL model crashes because it outputs zero steering values. Testing with improved reward weights showed **this is NOT a reward problem**. The root cause is a cascade of mismatches in observation space, input normalization, and network architecture.

**Most critical:** Input normalization mismatch between RL training and inference.

---

## Finding 1: Input Normalization Mismatch (🔴 CRITICAL)

### RL Observation Construction (`training/rl/gym_env.py:92-106`)
```python
obs = np.array([
    state.speed / 300.0,           # hardcoded scalar
    state.trackPos,                # already in [-1, 1]
    state.angle / np.pi,           # hardcoded scalar
    state.rpm / 10_000.0,          # hardcoded scalar
    float(self._gear) / 6.0,       # hardcoded scalar
    state.damage / 10_000.0,       # hardcoded scalar
    state.track[7] / 200.0,        # hardcoded scalar
    state.track[9] / 200.0,        # hardcoded scalar
    state.track[11] / 200.0,       # hardcoded scalar
])
```

**Method:** Per-sensor hardcoded divisors. NO z-score normalization.

### BC Observation Construction (`drivers/bc/driver.py:88-92` + `training/behavioral_cloning/dataset.py:49-55`)
```python
# During training:
self._sensors = (self._sensors - self._sensor_mean) / self._sensor_std

# At inference:
x = (x - self._mean) / self._std
```

**Method:** Z-score normalization using dataset statistics. The mean/std are learned from recorded telemetry and loaded from checkpoint.

### Why This Matters

**Scenario 1: High speed (180 km/h) on straight stretch**
- RL input: `[180/300, 0.0, 0.0, rpm/10k, ...]` = `[0.6, 0.0, 0.0, ...]`
- BC input (if speed ~120 km/h at training time): `[(180 - 80)/30, ...]` = `[3.3, ...]` (outlier, clipped by stats)

**Scenario 2: Sharp corner, need immediate steering**
- RL input: `[50/300, 0.5, 0.1, ...]` = `[0.17, 0.5, 0.1, ...]` (steering signal barely visible)
- BC input: `[(50 - 80)/30, 0.5, ...]` = `[-1.0, 0.5, ...]` (steering signal emphasized)

**Result:** The RL agent trained on raw scaled inputs where:
- Speed (0–1 scale) dominates the input vector
- Steering angle (also 0–1 scale) is visually similar but semantically opposite (forward vs. turn)
- The model learns to output zero steering because the optimization landscape is flatter for steering than for speed

---

## Finding 2: Track Sensor Index Mismatch

### RL uses indices: 7, 9, 11
```python
# gym_env.py:101-103
state.track[_TRACK_IDX[0]] / 200.0,  # track[7]  ← -6°
state.track[_TRACK_IDX[1]] / 200.0,  # track[9]  ← 0° (dead ahead)
state.track[_TRACK_IDX[2]] / 200.0,  # track[11] ← +6°
```

### BC uses indices: 6, 12, 18
```python
# dataset.py:15
SENSOR_COLS = ["speed", "trackPos", "angle", "rpm", "gear", "track_6", "track_12", "track_18"]

# driver.py:88-91
state.track[6], state.track[12], state.track[18]
```

### Interpretation
The TORCS `track` array has 19 elements. The indices correspond to different angles:
- Track[0..18]: angles from −90° to +90° (symmetric around center)
- Track[9]: dead ahead (center, 0°)
- Track[7]: ≈ −6°, Track[11]: ≈ +6° (narrow forward view)
- Track[6]: ≈ −9°, Track[12]: ≈ +9°, Track[18]: ≈ +81° (?)

**Issue:** If BC was trained on track[6, 12, 18] but RL uses [7, 9, 11], they see different lookahead geometry. BC has a wider view (±9°), RL has a narrower view (±6°).

---

## Finding 3: Network Architecture Inconsistency

### BC Model (`training/behavioral_cloning/model.py:11-42`)
```python
hidden_dims = [256, 256, 128]  # 3 hidden layers (default)
# Activations: ReLU + LayerNorm on each hidden layer
# Output heads: Tanh (steer), Sigmoid (accel, brake)
```

### RL Training Script (`training/rl/train_rl_bc_warmstart.py:78`)
```python
policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
```

### RL Driver at Inference (`drivers/rl/driver.py`)
- Uses `SB3.PPO.load()` which creates SB3's default `MlpPolicy`
- SB3 MlpPolicy default: `[64, 64]` if not specified
- **But train script specifies [256, 256]** → mismatch at load time!

**Problem:** If you trained with `[256, 256]` but the checkpoint's metadata says `[64, 64]`, the load will either:
1. Fail (architecture mismatch)
2. Load a truncated model (weights dropped)
3. Succeed but with frozen/broken weights

**Diagnosis:** Check `models/rl_bc_warmstart/model.zip` metadata:
```bash
python -c "
from stable_baselines3 import PPO
m = PPO.load('models/rl_bc_warmstart/model')
print(m.policy)
"
```

---

## Finding 4: Unused Damage Signal

### RL includes damage (9 features)
```python
state.damage / 10_000.0  # feature [5]
```

### BC ignores damage (8 features)
```python
SENSOR_COLS = ["speed", "trackPos", "angle", "rpm", "gear", "track_6", "track_12", "track_18"]
# no damage
```

**Why it matters:** Damage is sparse (zero until crash). Adding a mostly-zero feature increases input dimensionality without information gain. The extra feature can act as noise, especially with improper scaling.

---

## Finding 5: Output Scaling Asymmetry

### RL (`drivers/rl/driver.py:122-124`)
```python
steer = float(np.clip(action[0], -1.0, 1.0))
accel = float(np.clip(action[1], 0.0, 1.0))
brake = float(np.clip(action[2], 0.0, 1.0))
```

### BC (`drivers/bc/driver.py:109-112`)
```python
return Action(
    steer=float(out["steer"].item()),  # Tanh output already [-1, 1]
    accel=float(out["accel"].item()),  # Sigmoid output already [0, 1]
    brake=float(out["brake"].item()),  # Sigmoid output already [0, 1]
    gear=self._current_gear,
).clamp()
```

**Issue:** BC's output heads use activation functions that produce bounded outputs. RL's SB3 MlpPolicy uses tanh by default, but then you clip again — this is redundant but not harmful. However, if the network doesn't converge to learn steering, clipping won't help.

---

## Diagnosis: Why Model Outputs Zero Steering

1. **Normalization confuses the learned policy**
   - Model trained on `[0.6, 0.0, 0.0, ...]` scale inputs
   - Learned: "small steering changes in this scale don't matter much"
   - High-speed straight lines dominate training data
   - Steering gradient is noisy and small

2. **Observation mismatch with BC baseline**
   - BC was initialized from BC checkpoint, but observations don't match
   - BC checkpoint expects z-score normalized inputs; RL sends raw scaled inputs
   - The weight initialization carries assumptions from BC training (z-scored world) applied to RL training (raw-scaled world)

3. **Narrow lookahead (±6°) insufficient for Corkscrew**
   - Corkscrew has sharp corners
   - With only ±6° lookahead, the steering decision is made too late
   - Need at least ±15° or wider to plan ahead

4. **Network underspecification**
   - `[64, 64]` may be too small for 9-input problem with RL's noisy gradients
   - Compare: BC uses `[256, 256, 128]` trained on clean behavioral data

---

## Action Plan

### Phase 1: Verify the Mismatch (Do This First)

```bash
# Check RL model architecture
python -c "
from stable_baselines3 import PPO
m = PPO.load('models/rl_bc_warmstart/model', device='cpu')
print('Policy:', m.policy)
print('Network architecture:', m.policy.mlp_extractor.policy_net if hasattr(m.policy, 'mlp_extractor') else 'SB3 default')
"

# Check BC checkpoint
python -c "
import torch
ckpt = torch.load('models/bc_v2.pth', map_location='cpu', weights_only=False)
print('BC Input Dim:', ckpt['input_dim'])
print('BC Hidden Dims:', ckpt['hidden_dims'])
print('BC Sensor Mean:', ckpt['sensor_mean'])
print('BC Sensor Std:', ckpt['sensor_std'])
"
```

### Phase 2: Fix Observation Normalization (CRITICAL)

**Option A: Add z-score normalization to RL gym_env.py**

```python
# In gym_env.py, add computed statistics OR load from BC checkpoint:
# Method 1: Empirical stats from random rollouts (slower but automatic)
# Method 2: Load from BC checkpoint (faster, uses BC's data distribution)

def _make_obs(self, state: SensorState) -> np.ndarray:
    raw = np.array([
        state.speed,
        state.trackPos,
        state.angle,
        state.rpm,
        float(self._gear),
        state.damage,
        state.track[7],
        state.track[9],
        state.track[11],
    ], dtype=np.float32)
    
    # Z-score normalize using empirical or BC stats
    normalized = (raw - self._sensor_mean) / self._sensor_std
    return normalized
```

**Option B: Align RL observation space with BC**

```python
# Remove damage (not used by BC), use same track indices
def _make_obs(self, state: SensorState) -> np.ndarray:
    return np.array([
        state.speed / 300.0,
        state.trackPos,
        state.angle / np.pi,
        state.rpm / 10_000.0,
        float(self._gear) / 6.0,
        # REMOVED: state.damage / 10_000.0
        state.track[6] / 200.0,   # CHANGED: 7 → 6
        state.track[12] / 200.0,  # CHANGED: 9 → 12
        state.track[18] / 200.0,  # CHANGED: 11 → 18
    ], dtype=np.float32)
```

### Phase 3: Widen Lookahead (RECOMMENDED)

```python
# In gym_env.py, expand track indices to ±15°:
_TRACK_IDX = (4, 9, 14)  # instead of (7, 9, 11)
# This gives track sensors at ≈ -15°, 0°, +15°
```

### Phase 4: Retrain with Fixes

```bash
# After applying fixes, retrain RL from BC warm-start:
conda run -n ai_env python training/rl/train_rl_bc_warmstart.py \
    --bc-model models/bc_v2.pth \
    --target-steps 100000 \
    --sessions 60 \
    --save-path models/rl_bc_warmstart_v3_fixed
```

### Phase 5: Verify Steering Output

Before running a full lap, add debugging:

```python
# In drivers/rl/driver.py step(), before returning:
if self._step_count % 50 == 0:
    logger.info(
        "Step %d: obs=%s, action=%s, steer=%s, speed=%s",
        self._step_count, obs, action, steer, state.speed
    )
```

---

## Expected Outcome

- RL model should output **non-zero steering** on curves
- Lap time should improve compared to `rl_improved_reward_v1` (which crashes)
- Should be competitive with `rl_bc_warmstart` (50k steps) baseline

## Commits to Reference

- `56547e5`: Phase 3 complete (current best: 50k steps, working)
- `a8c0e68`: Improved reward (diagnosis showed NOT the issue)
- `5e6ed4d`: Failed test (zero-steering symptom identified)
- `10da2e2`: Reward reset (but observation/network issues remain)
