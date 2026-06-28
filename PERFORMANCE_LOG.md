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

| Model | Lap Time | Off-Track % | Status | Date Tested |
|-------|----------|-------------|--------|-------------|
| (Pending) | - | - | Testing... | 2026-06-28 22:13 |

---

**Target:** Match or exceed rule-based performance (148.448s lap)
