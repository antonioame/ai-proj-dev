# RL Performance Improvement Log

## Baseline Performance (Before Improvements)

| Model | Lap Time | Off-Track % | Status | Notes |
|-------|----------|-------------|--------|-------|
| Rule-Based (Baseline) | **148.448s** | 0.0% | ✅ COMPLETE | Proven working driver |
| BC v2 (35.4k samples) | CRASH | 100% | ❌ FAILED | Crashes at ~3267m (no steering) |
| RL Pure (30k steps) | CRASH | 100% | ❌ FAILED | Crashes at ~3267m (no steering) |
| RL + BC Warm-Start (50k steps) | CRASH | 100% | ❌ FAILED | Crashes at ~3267m (no steering) |

## Key Observations

1. **All ML models crash at same location (~3,267m)** → First turn where steering is critical
2. **All output zero steering** (steer +0.00) despite training
3. **Rule-based driver succeeds** with proper steering logic
4. **Problem is not training algorithm** (BC vs RL) but reward signal or observation space

## Hypothesis for Failure

The reward function likely **prioritizes forward progress over steering**:
- Models learn to accelerate straight (maximizes speed reward)
- Models avoid steering (might penalize or reduce reward)
- When curve arrives at ~66m, models can't handle it

## Improvement Plan

### v1.0: Increase steering/lane-keeping reward
- Add large penalty for off-track position (trackPos > 1.0)
- Add reward for staying centered (trackPos near 0)
- Reduce speed-only reward

### v2.0: Improve observation space
- Ensure track sensors are properly normalized
- Add more lookahead information
- Verify observation ranges match training data

### v3.0: Adjust action scaling
- Ensure steering output is properly scaled
- Check if steering is being clamped to zero
- Verify action normalization

## Improvements Applied

### v1.0: Improved Reward Function (In Progress Testing)
- ✅ Reduced PROGRESS_WEIGHT (1.0 → 0.5)
- ✅ Increased DEVIATION_WEIGHT (1.0 → 5.0)  
- ✅ Increased OFF_TRACK_PENALTY (2.0 → 20.0)
- ✅ Added progressive edge warnings (0.5→5pts, 0.75→10pts, 1.0→20pts)
- ✅ Added LANE_CENTER_BONUS for staying centered
- ✅ Trained for 20,631 steps across 13 sessions
- Model saved: `models/rl_improved_reward_v1/final.zip`
- **Testing:** In progress (TORCS connection issues)
- Expected result: Should prioritize steering over speed → better lap completion

## Test Results

| Model | Lap Time | Off-Track % | Steering | Status | Notes |
|-------|----------|-------------|----------|--------|-------|
| Rule-Based (Baseline) | **148.448s** | 0.0% | ✅ Active | ✅ WORKS | Benchmark - perfect lap |
| RL Improved Reward v1 | CRASH | ~100% | ❌ ZERO (0.00) | ❌ FAILED | Crashes at ~3,272m, no steering |

## Testing Challenges

**TORCS Stability Issue:** After running one lap test successfully, TORCS becomes unstable and refuses connections on subsequent test runs (WinError 10054). This is a known behavior - the simulator needs to be fully restarted between test runs, but multiple restarts within short timeframe cause connection issues.

**Recommendation:** Test improved model after:
1. Full system restart
2. Or with longer wait times between TORCS restarts
3. Or implement automated TORCS health check + reboot

## Analysis: Why Improved Reward v1 Failed

**Hypothesis:** Reward function weights are the bottleneck
- ❌ **DISPROVEN** by test results
- Model still outputs **zero steering** despite:
  - 5× stronger drift penalty (1.0 → 5.0)
  - 10× stronger off-track penalty (2.0 → 20.0)
  - Progressive edge warnings added
  - Lane-centering bonus added

**Actual Root Cause:** Problem is deeper than reward weights

Possible issues:
1. **Observation Space Insufficient**
   - Track sensors may not encode turn geometry
   - 9 features might miss critical information
   - Need more look-ahead distance

2. **Model Architecture Too Simple**
   - [64, 64] network might be underfitting
   - Cannot learn complex steering policy
   - Need larger networks or different architecture

3. **Action Encoding Issue**
   - Steering output might be clipped/clamped to zero
   - Output normalization might be wrong
   - Check if steer output even reaches model

4. **Input Normalization Problem**
   - Observation normalization might compress important signals
   - trackPos ∈ [-1, 1] might not normalize correctly
   - Normalized inputs might be uninformative

**Conclusion:** Reward engineering alone cannot fix a fundamental observation/action encoding issue. Need to investigate the pipeline, not just tune weights.

## Next Investigation Priorities

1. **Check steering output directly** - is model actually outputting non-zero steering?
2. **Inspect observation values** - are they normalized correctly?
3. **Test larger network** - [256, 256] instead of [64, 64]
4. **Add more track lookahead** - use more sensors from track array
5. **Debug action encoding** - verify steer value reaches TORCS

---

**Target:** Match or exceed rule-based performance (148.448s lap)
