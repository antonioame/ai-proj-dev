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

| Model | Lap Time | Off-Track % | Status | Notes |
|-------|----------|-------------|--------|-------|
| Rule-Based (Baseline) | **148.448s** | 0.0% | ✅ CONFIRMED | Benchmark - perfect lap |
| RL Improved Reward v1 | UNTESTED | - | ⏳ BLOCKED | TORCS connection unstable after first test |

## Testing Challenges

**TORCS Stability Issue:** After running one lap test successfully, TORCS becomes unstable and refuses connections on subsequent test runs (WinError 10054). This is a known behavior - the simulator needs to be fully restarted between test runs, but multiple restarts within short timeframe cause connection issues.

**Recommendation:** Test improved model after:
1. Full system restart
2. Or with longer wait times between TORCS restarts
3. Or implement automated TORCS health check + reboot

## Theoretical Improvement Analysis

Despite testing blockers, the reward function improvements are sound:

**Original Reward Issues:**
- Speed reward (1.0×) competed with drift penalty (speed × trackPos)
- Off-track penalty only triggered AFTER crash (trackPos > 1.0)
- Model learned: "accelerate straight, don't worry about curves"

**Improved v1 Fixes:**
- ✅ Speed reward reduced 50% (1.0 → 0.5)
- ✅ Drift penalty 5× stronger (1.0 → 5.0)
- ✅ Off-track penalty 10× stronger (2.0 → 20.0)
- ✅ Early warnings at 0.5 and 0.75 position
- ✅ Lane-centering bonus (positive reinforcement)

**Expected Outcome:** Model should now:
- Prioritize staying on-track over speed
- Learn to steer to avoid edges
- Potentially complete full lap (vs crash at ~66m)

---

**Target:** Match or exceed rule-based performance (148.448s lap)
