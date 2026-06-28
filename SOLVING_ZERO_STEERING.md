# Solving Zero-Steering: Complete Summary

**Status:** ✅ All fixes applied and committed (commit `727593b`)

---

## What Was Fixed

### Root Cause: Input Normalization Mismatch

The RL model was trained on **raw scaled inputs** (speed/300, rpm/10000, etc.) while BC was trained on **z-score normalized inputs** learned from actual telemetry data. This caused:
- Steering signal suppressed during optimization
- Model output: zero steering on all curves
- Result: crash at ~3.2 km

### The Three Critical Fixes

#### Fix 1: Observation Space Alignment (gym_env.py)
```python
# Before: 9 features, raw scaled, track[7,9,11]
_TRACK_IDX = (7, 9, 11)   # ±6° lookahead
OBS_DIM = 9
obs = [speed/300, trackPos, angle/π, rpm/10k, gear/6, damage/10k, track[7]/200, ...]

# After: 8 features, z-score normalized, track[6,12,18]
_TRACK_IDX = (6, 12, 18)  # ±9° lookahead (wider)
OBS_DIM = 8
_OBS_MEAN = [88.33, -0.07, 0.008, 6952, 3.97, 48.5, 43.3, 8.35]  # from BC v2.pth
_OBS_STD = [38.33, 0.30, 0.05, 1590, 1.45, 37.0, 31.1, 5.26]     # from BC v2.pth
obs = (raw - _OBS_MEAN) / _OBS_STD  # match BC training exactly
```

#### Fix 2: RL Driver Alignment (drivers/rl/driver.py)
- Import `_OBS_MEAN`, `_OBS_STD`, `_TRACK_IDX` from gym_env
- Use identical `_make_obs()` implementation
- **Guarantees:** training/inference symmetry (no surprises at test time)

#### Fix 3: Training Script Overhaul (train_rl_bc_warmstart.py)
```
BEFORE: Start TORCS → Create model (timeout fires mid-creation)
AFTER:  Create model → Start TORCS (ready before handshake)

BEFORE: Step count = int(600 * elapsed_time)  (fake)
AFTER:  Step count from SB3 callback (real)

BEFORE: BC weights never initialized (0 matched)
AFTER:  Properly map BC backbone → PPO policy_net (4 parameter tensors)
```

---

## Changes Committed (commit 727593b)

### Code Changes
- `training/rl/gym_env.py`: Observation normalization, track indices, 8→9 feature count
- `drivers/rl/driver.py`: Import and use shared normalization
- `training/rl/train_rl_bc_warmstart.py`: Complete rewrite for correct startup order

### Documentation (5 new guides)
- `INVESTIGATION_REPORT.md` — Root cause diagnosis summary
- `docs/ZERO_STEERING_SUMMARY.md` — Executive summary
- `docs/INVESTIGATION_ZERO_STEERING.md` — Technical deep-dive (5 root causes)
- `docs/FIX_ZERO_STEERING.md` — Implementation plan with code
- `docs/RUNNING_DRIVERS.md` — How to run 10+ models

---

## Verification

All checks passed:
- ✅ 37 unit tests pass
- ✅ Observation shape: 8 (matches BC)
- ✅ Track indices: (6, 12, 18) (matches BC)
- ✅ Normalization: z-score with BC stats
- ✅ gym_env and RL driver observations identical
- ✅ PPO model builds without TORCS
- ✅ BC v2 checkpoint input dim verified: 8

---

## Next Steps to Retrain

### 1. Stop TORCS and any running Python processes (if needed)

### 2. Retrain with fixed observation space

```bash
conda run -n ai_env python training/rl/train_rl_bc_warmstart.py \
    --bc-model models/bc_v2.pth \
    --target-steps 100000 \
    --sessions 120 \
    --save-path models/rl_bc_warmstart_v3_fixed
```

**Timing:** ~2-3 hours (120 sessions × ~1 minute overhead each)
**Output:** `models/rl_bc_warmstart_v3_fixed/final.zip`

### 3. Test on one lap

```bash
conda run -n ai_env python scripts/run_agent.py \
    --driver rl_rl_bc_warmstart_v3_fixed \
    --laps 1 \
    --telemetry
```

**Expected results:**
- ✅ Steering non-zero on curves (not all zeros)
- ✅ Lap completes (no crash at 3.2 km)
- ✅ Lap time < 150 s (goal: competitive with rule-based 148.4 s)

### 4. Compare performance

```bash
# Baseline (should complete ~148 s)
conda run -n ai_env python scripts/run_agent.py --driver rule_based --laps 1

# Improved RL (target: < 150 s)
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v3_fixed --laps 1
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Load BC stats hardcoded instead of from checkpoint | Avoids PyTorch dependency in driver, faster inference |
| 8 features instead of 9 (remove damage) | Damage is sparse (0 until crash); dilutes steering signal |
| Track [6,12,18] instead of [7,9,11] | Wider lookahead (±9° vs ±6°) better for Corkscrew corners |
| Z-score normalize in both gym_env and driver | Guarantees symmetry; if BC stats change, update one place |
| Build model before TORCS starts | Eliminates pre-connection timeout (was fatal issue) |

---

## Troubleshooting

### If steering is still zero after retraining:

1. **Verify normalization is applied:**
   ```bash
   python -c "from training.rl.gym_env import _OBS_MEAN, _OBS_STD; print(_OBS_MEAN, _OBS_STD)"
   ```
   Should print BC stats, not zeros.

2. **Check BC checkpoint loaded:**
   ```bash
   python -c "import torch; c=torch.load('models/bc_v2.pth', map_location='cpu', weights_only=False); print(c['input_dim'])"
   ```
   Should print 8.

3. **Examine telemetry CSV:**
   - Check that `steer` column has non-zero values on curves
   - Check that `speed` is sensible (not all 0 or all 300)

4. **Try wider lookahead:**
   - Change `_TRACK_IDX = (6, 12, 18)` to `(4, 9, 14)` for ±15° view
   - Retrain to see if wider vision helps steering

### If training still times out:

- Increase `STEPS_PER_SESSION` in train script gradually (currently 1000)
- Check TORCS process is actually running: `tasklist | grep wtorcs`
- Verify localhost:3001 port is free: `netstat -ano | grep 3001`

---

## Rollback

If something breaks:
```bash
git revert HEAD  # Reverts commit 727593b
git checkout HEAD~1 -- training/rl/gym_env.py drivers/rl/driver.py
```

Old `rl_bc_warmstart/final.zip` (50k steps) remains valid baseline.

---

## Performance Expectations

| Model | Steering | Lap Time | Notes |
|-------|----------|----------|-------|
| Rule-based | ✅ Full | 148.4 s | Baseline |
| BC v1/v2 | ✅ Full | ? | Behavioral cloning |
| RL bc_warmstart (old) | ❌ Zero | Crash ~3.2km | Input mismatch |
| RL bc_warmstart_v3_fixed | ✅ Full | ? | After fix (to be tested) |

**Goal:** `v3_fixed` lap time **< 150 seconds** to match rule-based performance.

---

## Files Modified

```
training/rl/gym_env.py              ← Observation space fix
drivers/rl/driver.py                ← Inference symmetry
training/rl/train_rl_bc_warmstart.py ← Training script rewrite
INVESTIGATION_REPORT.md             ← New (summary)
docs/ZERO_STEERING_SUMMARY.md       ← New
docs/INVESTIGATION_ZERO_STEERING.md ← New
docs/FIX_ZERO_STEERING.md          ← New
docs/RUNNING_DRIVERS.md            ← New
```

All changes backward-compatible with existing tests.

---

## Commit Info

- **Hash:** 727593b
- **Author:** Claude Sonnet 4.6
- **Message:** Fix zero-steering root cause: align RL observation space with BC training
- **Changes:** 11 files, 1342 insertions, 147 deletions
