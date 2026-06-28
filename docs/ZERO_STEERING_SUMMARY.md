# Zero-Steering Issue — Executive Summary

**Status:** Root cause identified and fix plan created.

---

## TL;DR

The RL model crashes after ~3.2 km because it **always outputs zero steering**, even on sharp curves.

**Why:** The model was trained on raw scaled inputs `[speed/300, trackPos, angle/π, ...]` but should have been trained on z-score normalized inputs like BC was. This makes the input distribution incompatible with BC's learned weights and breaks the optimization landscape for steering.

**How to fix:** Align RL observation space with BC (same 8 features, z-score normalization, same track indices).

---

## Commit Timeline

| Commit | Event |
|--------|-------|
| `56547e5` | Phase 3 complete: PPO trained 50k steps, BC warm-start. **Baseline works.** |
| `a8c0e68` | Reward overhaul attempt: 5× drift penalty, 10× off-track, lane-center bonus |
| `5e6ed4d` | Testing failed: model still crashes, **outputs zero steering.** Realized NOT a reward problem. |
| `10da2e2` | Reset reward.py to working state — but observation/network issues remain unfixed. |

---

## Root Cause Analysis

### ✅ Verified Issues (by actual model inspection)

1. **Input normalization mismatch** 🔴 CRITICAL
   - BC: z-score normalized using dataset stats (mean/std learned from 35k samples)
   - RL: hardcoded per-sensor divisors (speed/300, rpm/10000, etc.)
   - Result: RL sees speed 150 km/h as "normal" (0.5 scale), steering also 0–1 scale → model learns they're equivalent → steering suppressed

2. **Track sensor index mismatch** 🔴 CRITICAL
   - BC uses indices: [6, 12, 18]
   - RL uses indices: [7, 9, 11]
   - Result: RL has narrower lookahead for Corkscrew's sharp corners

3. **Network architecture divergence** 🟠 HIGH
   - BC: [256, 256, 128] + LayerNorm + ReLU (3 hidden layers)
   - RL: [256, 256] + Tanh only (2 hidden layers)
   - Result: RL's simpler network + less stable activation doesn't learn steering well

4. **Damage feature noise** 🟡 MEDIUM
   - RL includes damage (9 features), BC doesn't (8 features)
   - Result: extra sparse feature dilutes steering signal

---

## Documentation

Three new docs created in `docs/`:

### 1. `RUNNING_DRIVERS.md`
**What:** Quick reference for running different drivers and models.

**When to read:** Before running any lap. Lists all 10 trained models and how to invoke them.

### 2. `INVESTIGATION_ZERO_STEERING.md`
**What:** Deep-dive into each root cause with evidence and impact analysis.

**When to read:** To understand WHY the model fails. See "Finding 1" through "Finding 5" for detailed mechanics.

### 3. `FIX_ZERO_STEERING.md`
**What:** Implementation plan with actual code changes, step-by-step.

**When to read:** Before implementing the fix. Contains Python code snippets to apply to gym_env.py and rl/driver.py.

---

## Action: Implement the Fix

### Before You Start

Make sure TORCS is NOT running. We're modifying training code, not running the agent yet.

### Step 1: Apply Observation Space Fix

**File:** `training/rl/gym_env.py`

- Update `_make_obs()` to use 8 features instead of 9 (remove damage)
- Change track indices from [7, 9, 11] to [6, 12, 18] to match BC
- Add z-score normalization loading from BC checkpoint

**Full code:** See `FIX_ZERO_STEERING.md` Phase 1 and Step 1.

### Step 2: Update RL Driver

**File:** `drivers/rl/driver.py`

- Update `_make_obs()` to match gym_env exactly
- Use same normalization stats

**Full code:** See `FIX_ZERO_STEERING.md` Step 2.

### Step 3: Retrain RL from BC Warm-Start

```bash
conda run -n ai_env python training/rl/train_rl_bc_warmstart.py \
    --bc-model models/bc_v2.pth \
    --target-steps 100000 \
    --sessions 60 \
    --save-path models/rl_bc_warmstart_v3_fixed
```

Expected: 60 TORCS sessions, ~1–2 hours, saves to `models/rl_bc_warmstart_v3_fixed/final.zip`.

### Step 4: Test the Fixed Model

```bash
conda run -n ai_env python scripts/run_agent.py \
    --driver rl_rl_bc_warmstart_v3_fixed \
    --laps 1 \
    --telemetry
```

**What to look for:**
- Steering values change on curves (not all zeros)
- Lap completes (doesn't crash at ~3.2 km)
- Lap time < 150 s (baseline rule-based: 148.4 s)

---

## Expected Outcomes

### After Fix

| Aspect | Before | After (Expected) |
|--------|--------|------------------|
| Steering output | All zeros on curves | Non-zero, follows curves |
| Model crash | ~3.2 km | Full lap completion |
| Lap time | N/A (crash) | < 150 s (competitive) |
| Input alignment | RL ≠ BC | RL = BC ✓ |

### Success Metrics

- ✅ Model outputs **non-zero steering** during Corkscrew turns
- ✅ **Completes full lap** without crashing
- ✅ Lap time **< 150 seconds** (within striking distance of rule-based)

### What if it still doesn't work?

If steering is still zero after the fix:
1. Add logging to `_make_obs()` to verify normalization is applied
2. Check that BC checkpoint loaded successfully (fallback values are used otherwise)
3. Verify track indices are correct by logging state.track[6], [12], [18] values
4. Consider network enhancement: use [256, 256, 128] architecture (see Phase 3 of FIX_ZERO_STEERING.md)

---

## Architecture Comparison

### Behavioral Cloning (BC v2) — Currently Working

```
Input (8): [speed, trackPos, angle, rpm, gear, track[6], track[12], track[18]]
           ↓ Z-score normalize (learned from 35k samples)
Backbone:  [256 → ReLU+LN] → [256 → ReLU+LN] → [128 → ReLU+LN]
           ↓
Heads:     steer(Tanh) + accel(Sigmoid) + brake(Sigmoid)
Output: Actions [−1,1] × [0,1] × [0,1]
```

### RL before fix (Broken)

```
Input (9): [speed, trackPos, angle, rpm, gear, damage, track[7], track[9], track[11]]
           ↓ Hardcoded divisors (speed/300, rpm/10k, etc.) — NO normalization
MLP:       [256 → Tanh] → [256 → Tanh]
           ↓
Action:    [steer, accel, brake] — clipped to bounds
```

**Problem:** Input distribution mismatch breaks BC warm-start initialization.

### RL after fix (Expected)

```
Input (8): [speed, trackPos, angle, rpm, gear, track[6], track[12], track[18]]
           ↓ Z-score normalize (same as BC)
MLP:       [256 → Tanh] → [256 → Tanh]
           ↓
Action:    [steer, accel, brake]
```

**Now:** Input matches BC training distribution → warm-start is meaningful → steering learned.

---

## Rollback Safety

If the fix breaks something:

```bash
git checkout HEAD~1 -- training/rl/gym_env.py drivers/rl/driver.py
```

The `rl_bc_warmstart/final.zip` checkpoint (50k steps) remains valid and is still runnable.

---

## References

- `docs/RUNNING_DRIVERS.md` — How to run drivers
- `docs/INVESTIGATION_ZERO_STEERING.md` — Why the model fails
- `docs/FIX_ZERO_STEERING.md` — How to fix it (with code)
- Commit `56547e5` — Last working RL baseline
