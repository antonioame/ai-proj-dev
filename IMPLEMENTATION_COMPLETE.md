# Zero-Steering Fix: Implementation Complete ✓

**Date:** 2026-06-29  
**Status:** All fixes applied, model trained, ready for testing

---

## What Was Done

### 1. Root Cause Investigation
Identified the zero-steering problem was caused by **input normalization mismatch** between RL training and BC:
- RL trained on: hardcoded divisors `[speed/300, rpm/10k, ...]`
- BC trained on: z-score normalized `[(speed - mean) / std, ...]`
- Result: Model couldn't learn steering

### 2. Code Fixes Applied (Commit 727593b)

#### gym_env.py
- Changed observation space from 9 → 8 features (removed damage)
- Changed track indices from [7, 9, 11] → [6, 12, 18] (matches BC)
- Added z-score normalization using BC v2.pth statistics
- New normalization constants hardcoded for consistency

#### drivers/rl/driver.py
- Import shared normalization constants from gym_env
- Use identical `_make_obs()` implementation
- Guarantees training/inference symmetry

#### training/rl/train_rl_bc_warmstart.py
- **Complete rewrite** with 3 major improvements:
  1. **Build model BEFORE starting TORCS** (fixes pre-connection timeout)
  2. **Proper BC weight initialization** (map BC backbone → PPO policy_net)
  3. **Real step counting** (via SB3 callback, not elapsed-time estimates)
- Steps per session: 1000 (low enough to avoid TORCS timeout)

### 3. Testing
- ✅ All 37 unit tests pass
- ✅ Observation alignment verified (gym_env = RL driver)
- ✅ BC checkpoint validation (8-dim, stats match)
- ✅ PPO model builds without TORCS

### 4. Training Executed
```
Training: RL Fine-tuning with BC Warm-start (v3_fixed)
  Start time: 2026-06-29 00:28:28
  End time:   2026-06-29 00:35:46
  Duration:   ~7.3 minutes
  Sessions:   37
  Total steps: 100,488 (target: 100,000)
  Output:     models/rl_bc_warmstart_v3_fixed/final.zip (1.6 MB)
```

---

## What to Test Next

### Option A: Single Lap (Quick Test)

```bash
# Start TORCS
wtorcs.exe -r torcs_env\race_config\corkscrew_solo.xml

# Run model on 1 lap with telemetry
conda run -n ai_env python scripts/run_agent.py ^
    --driver rl_rl_bc_warmstart_v3_fixed ^
    --laps 1 ^
    --telemetry
```

**Expected:**
- Steering values **non-zero** on curves (not all zeros)
- Lap **completes** (no crash at ~3.2 km)
- Lap time recorded in `results/`

**Check telemetry CSV for:**
```
Column "steer": should vary during curves (not constant 0)
Column "speed": should show realistic speeds
Row count: should be ~5000-10000 (full lap)
```

### Option B: Performance Comparison (5 laps)

```bash
# Baseline (rule-based): ~148 s/lap
conda run -n ai_env python scripts/run_agent.py --driver rule_based --laps 5

# Improved RL (target: < 150 s/lap)
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v3_fixed --laps 5
```

---

## Key Metrics

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Steering output | Zero on curves | Non-zero, progressive |
| Lap completion | Crash at 3.2 km | Full lap |
| Observation dim | 9 (raw scaled) | 8 (z-scored) |
| Track lookahead | ±6° narrow | ±9° wider |
| Model size | — | 1.6 MB |
| Training time | — | ~7 minutes (100k steps) |

---

## Files Changed

```
training/rl/gym_env.py              (observation space fix)
drivers/rl/driver.py                (inference alignment)
training/rl/train_rl_bc_warmstart.py (training pipeline rewrite)
INVESTIGATION_REPORT.md             (root cause doc)
docs/ZERO_STEERING_SUMMARY.md       (executive summary)
docs/INVESTIGATION_ZERO_STEERING.md (technical deep-dive)
docs/FIX_ZERO_STEERING.md          (implementation guide)
docs/RUNNING_DRIVERS.md            (how-to for all models)
SOLVING_ZERO_STEERING.md           (this approach doc)
```

---

## Trained Model Location

```
models/rl_bc_warmstart_v3_fixed/
├── model.zip     [1.6 MB] - Latest checkpoint (session 37)
└── final.zip     [1.6 MB] - Exported final checkpoint
```

**Run with:**
```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v3_fixed --laps 1
```

---

## Verification Checklist

- [x] Root cause identified (input normalization mismatch)
- [x] Code fixes applied (gym_env, driver, training script)
- [x] All unit tests pass (37/37)
- [x] Observation alignment verified
- [x] Model trained (100k+ steps)
- [x] Checkpoint saved
- [ ] Model tested on 1 lap (YOU NEXT)
- [ ] Steering confirmed non-zero (YOU NEXT)
- [ ] Lap time measured (YOU NEXT)

---

## If It Still Doesn't Work

### Symptom: Steering still zero

1. **Check observation normalization:**
   ```bash
   python -c "from training.rl.gym_env import _OBS_MEAN; print(_OBS_MEAN)"
   ```
   Should print BC stats (88.3, -0.07, 0.008, ...), not zeros.

2. **Check track indices:**
   ```bash
   python -c "from training.rl.gym_env import _TRACK_IDX; print(_TRACK_IDX)"
   ```
   Should print (6, 12, 18), not (7, 9, 11).

3. **Look for logs:**
   - Check `results/rl_rl_bc_warmstart_v3_fixed_*.json` for lap time
   - Check any `.csv` telemetry for steer column values

4. **Try alternate model:**
   ```bash
   # Fallback to previous best (50k steps)
   conda run -n ai_env python scripts/run_agent.py --driver rl_bc_warmstart --laps 1
   ```

---

## Success Criteria

Model is working if:

1. ✅ **Steering non-zero:** Steer column in telemetry CSV has values like 0.1, -0.2, 0.05 (not all 0)
2. ✅ **Lap completes:** Model runs for ~5000+ steps without crash
3. ✅ **Lap time < 150 s:** Performance within striking distance of rule-based (148.4 s)

---

## Next Phase (After Testing)

Once steering works:

1. **Measure lap time gain:**
   - Compare v3_fixed vs rule_based over 5 laps
   - Target: within 2-3% of rule-based (< 153 s)

2. **Consider refinements:**
   - Wider track lookahead: change `_TRACK_IDX = (4, 9, 14)` for ±15°
   - Larger network: try `[512, 512, 256]` if steering is weak
   - More training: extend to 200k steps if still improving

3. **Document results:**
   - Save best lap time to `PERFORMANCE_LOG.md`
   - Record which driver variant performed best

---

## Commit Reference

- **Commit:** 727593b
- **Branch:** main2
- **Date:** 2026-06-29
- **Changes:** 11 files modified, 1342 insertions

## Training Reference

- **Script:** train_rl_bc_warmstart.py (rewritten)
- **BC warm-start:** models/bc_v2.pth
- **Observation space:** 8 features, z-score normalized
- **Steps trained:** 100,488 (37 sessions)
- **Duration:** ~7.3 minutes
- **Model:** models/rl_bc_warmstart_v3_fixed/final.zip
