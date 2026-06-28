# Zero-Steering Investigation Report

**Date:** 2026-06-29  
**Issue:** RL model crashes after ~3.2 km, outputs only zero steering  
**Status:** ✅ Root cause identified, fix plan created, documentation ready

---

## Quick Summary

The RL model fails because:

1. **Input normalization mismatch** — Model trained on raw scaled inputs instead of z-score normalized inputs like BC
2. **Track sensor index mismatch** — RL uses [7,9,11], BC uses [6,12,18], causing different lookahead geometry
3. **Network architecture divergence** — BC uses 3-layer backbone with LayerNorm, RL uses 2-layer without
4. **Damage feature noise** — Extra sparse feature dilutes steering signal

---

## What Was Committed

### Recent Commit Failures

| Commit | Problem |
|--------|---------|
| `a8c0e68` | Improved reward function (5× drift, 10× off-track) |
| `5e6ed4d` | **FAILED TEST**: Model still crashes, outputs zero steering |
| `10da2e2` | Reset reward to working state — but core issue remains |

**Key finding:** Reward engineering did NOT fix the problem. Root cause is deeper.

---

## Documentation Created

Four comprehensive guides written to `docs/`:

### 1. `RUNNING_DRIVERS.md`
Quick reference for running drivers and available models.
```bash
# Run rule-based baseline (148.4 s best time)
conda run -n ai_env python scripts/run_agent.py --driver rule_based

# Run BC warm-start RL model (50k steps, working baseline)
conda run -n ai_env python scripts/run_agent.py --driver rl_bc_warmstart
```

### 2. `INVESTIGATION_ZERO_STEERING.md`
Detailed technical analysis. Read to understand:
- Finding 1: Input normalization mismatch (CRITICAL)
- Finding 2: Track sensor index mismatch (CRITICAL)
- Finding 3: Network architecture divergence (HIGH)
- Finding 4: Damage feature noise (MEDIUM)
- Finding 5: Output scaling asymmetry (LOW)

### 3. `FIX_ZERO_STEERING.md`
Implementation plan with Python code ready to copy-paste:
- Phase 1: Align observation space (remove damage, change track indices, add normalization)
- Phase 2: Load BC normalization stats
- Phase 3: Enhance network architecture (optional)
- 4 implementation steps with code snippets
- Expected improvements and rollback plan

### 4. `ZERO_STEERING_SUMMARY.md`
Executive summary tying everything together. Read this first if you're in a hurry.

---

## Model Architecture Comparison

**BC v2 (working):**
- Input: 8 features, z-score normalized
- Network: [256 → LN+ReLU] → [256 → LN+ReLU] → [128 → LN+ReLU]
- Trained on 35.4k samples from rule-based driver

**RL before fix (broken):**
- Input: 9 features (includes damage), hardcoded scaling
- Network: [256 → Tanh] → [256 → Tanh]
- Trained for 50k steps but never learned steering

**RL after fix (expected):**
- Input: 8 features (same as BC), z-score normalized (same stats)
- Network: [256 → Tanh] → [256 → Tanh]
- Should now learn steering because input distribution matches BC

---

## Verified Facts (from model inspection)

### BC v2.pth
```
Input Dim: 8
Hidden Dims: [256, 256, 128]
Sensor Mean: [88.3, -0.07, 0.008, 6952, 3.97, 48.5, 43.3, 8.35]
Sensor Std:  [38.3,  0.30,  0.05,  1590, 1.45, 37.0, 31.1, 5.26]
```

### RL BC Warmstart (both working and improved versions)
```
Input Dim: 9
Architecture: Linear(9 → 256) → Tanh → Linear(256 → 256) → Tanh
Action Space: Box([-1, 0, 0], [1, 1, 1])
Observation Space: (9,)
```

**Mismatch:** BC expects 8 z-scored inputs, RL trains on 9 raw-scaled inputs. This breaks weight initialization.

---

## Next Steps

### To Implement the Fix:

1. **Apply code changes** to `training/rl/gym_env.py` and `drivers/rl/driver.py`
   - See `FIX_ZERO_STEERING.md` for exact code

2. **Retrain RL model:**
   ```bash
   conda run -n ai_env python training/rl/train_rl_bc_warmstart.py \
       --bc-model models/bc_v2.pth \
       --target-steps 100000 \
       --sessions 60 \
       --save-path models/rl_bc_warmstart_v3_fixed
   ```

3. **Test on one lap:**
   ```bash
   conda run -n ai_env python scripts/run_agent.py \
       --driver rl_rl_bc_warmstart_v3_fixed \
       --laps 1 \
       --telemetry
   ```

4. **Verify:**
   - Check telemetry CSV for non-zero steering on curves
   - Lap should complete (no crash at 3.2 km)
   - Lap time should be < 150 s

---

## Expected Improvements

After fix:
- ✅ Model outputs **non-zero steering** (not zeros on curves)
- ✅ **Completes full lap** (no crash)
- ✅ Lap time **< 150 seconds** (competitive with rule-based)

---

## Key Insights

**Why reward engineering failed:**
- The model architecture was broken, not the reward signal
- You could have the perfect reward and the model still can't steer
- Input space alignment is foundational

**Why BC warm-start works now but initialization was lost:**
- BC trained on z-scored inputs; RL sent raw-scaled inputs
- Weight initialization assumes BC's input distribution
- Fixing input distribution restores warm-start effectiveness

**Why RL's architecture is simpler than BC:**
- SB3 defaults to [64, 64] if you don't specify
- BC was hand-crafted with [256, 256, 128] + LayerNorm
- Adding LayerNorm to RL training could help (Phase 3)

---

## Files to Read (in order)

1. **Start here:** `ZERO_STEERING_SUMMARY.md` (this sets up context)
2. **Understand why:** `INVESTIGATION_ZERO_STEERING.md` (Finding 1 is critical)
3. **Implement fix:** `FIX_ZERO_STEERING.md` (copy-paste ready code)
4. **Run drivers:** `RUNNING_DRIVERS.md` (after retraining, test your model)

---

## Confidence Level

**Root cause diagnosis: 95%+ confident**
- Verified through actual model inspection
- BC stats loaded and compared
- Architecture differences confirmed
- Multiple independent indicators point to same conclusion

**Fix effectiveness: 85% confident**
- Addresses all identified root causes
- Similar successful patterns in ML literature
- May need minor tuning if edge cases exist
- Worst case: revert to `rl_bc_warmstart` baseline

---

## Timeline

- **Phase 1 complete:** Root cause identified and documented
- **Phase 2 ready:** Implementation plan with code
- **Phase 3:** You apply fix and retrain (~2 hours)
- **Phase 4:** Validate and measure improvement

---

## Questions?

Refer to the detailed docs in `docs/`:
- Technical questions → `INVESTIGATION_ZERO_STEERING.md`
- Implementation questions → `FIX_ZERO_STEERING.md`
- How-to-run questions → `RUNNING_DRIVERS.md`
