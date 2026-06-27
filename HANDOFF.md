# Handoff: Corkscrew Lap-Time Optimization (Session 2026-06-27)

## Summary
**Phase B is COMPLETE and COMMITTED.** ABS braking system + higher brake limits achieved **−3.24 seconds** (151.7s → 148.4s).

**Phase C (OptimalLineDriver)** is drafted but BUGGY—the car crashes at 480m complex. Needs steering/control debugging.

---

## What Was Done

### Phase A (Instrumentation) — PARTIAL ✓
- ✅ Measured **track length** = 3608.4 m (via telemetry lap-end distFromStart reset)
- ✅ Built `torcs_env/track_map.py`: data structure for per-bucket terrain (speed, curvature, trackPos)
- ✅ Built `scripts/build_track_map.py`: compile telemetry CSV → track_map.json (5m bucket resolution)
- ✅ Enhanced `run_agent.py` telemetry: now captures distFromStart, curLapTime, wheelSpinVel, damage
- ✅ Created `laptime_ledger.csv`: single source of truth for all optimization runs
- ✅ Built `scripts/benchmark.py`: run K laps, append metrics to ledger, compare vs baseline
- ⚠️  Track map has 84% buckets marked as "corner" (curvature threshold too low for clean straight/corner splits—not critical for this phase)

### Phase B (Quick Wins) — COMPLETE ✓
- ✅ **Anti-lock Braking System (ABS)**
  - Added `_apply_abs()` method to RuleBasedDriver
  - Detects front wheel lock (wheel spin < 80% of ground speed)
  - Reduces brake pressure proportionally: `brake × (1 − lockup / threshold)`
  - Allows BRAKE_MAX to be raised safely (was limited by lockup risk)

- ✅ **Higher Brake Pressure Limits**
  - BRAKE_MAX_HIGH: 0.65 → **0.82** (>140 km/h)
  - BRAKE_MAX_MED: 0.78 → **0.88** (90–140 km/h)
  - BRAKE_MAX_LOW: 0.90 → **0.93** (<90 km/h)
  - BRAKE_DECEL_FACTOR: 255 → **270** (reflects ~1.05 g deceleration with ABS)

- ✅ **Result**: 151.688 s → **148.448 s** (−3.240 s, −2.1%)

### Phase C (OptimalLineDriver) — BUGGY ⚠️
Files drafted but NOT COMMITTED:
- `drivers/optimal/driver.py` — position-indexed trajectory controller
- `drivers/optimal/trajectory.py` — speed profile builder (corner→MAX_SPEED backward pass)
- `torcs_env/track_data/track_map.json` — built from baseline telemetry
- `scripts/build_track_map.py` — track map builder

**The bug**: OptimalLineDriver crashes at 480m (distFromStart) complex in recovery loop.
- Trajectory speed profile looks correct (min 35 km/h at apex, 38–48 km/h through the complex)
- Car goes flat-out, then brakes hard (as designed) — but steering fails
- Result: car ends up at trackPos = −7.4 (massively off-track), speed 0.2 km/h, stuck forever
- Log: repeats recovery action (0.3 accel, steer=±0.30) every ~20 ms

**Root cause suspects** (in priority order):
1. **Steering in OptimalLineDriver._steer()** may not be strong enough through the complex
   - Line gain too low (0.20)? Angle gain too low (2.0)?
   - Missing apex-seeking like rule_based driver uses
2. **Startup phase timing** — OptimalLineDriver uses STARTUP_STEPS=80, but trajectory might not account for launch dynamics
3. **Track map `distFromStart` lookup** — off by one? Wrapping issue near finish line?

---

## Current Baseline & Ledger

```
timestamp            | config_id                    | best_lap_s | delta_vs_baseline
2026-06-27T16:18:23  | baseline_rule_based          | 151.688    | baseline
2026-06-27T16:38:29  | phase_b_abs_higher_brakes    | 148.448    | −3.240s (−2.1%)
```

Run with:
```bash
# Phase B (current best)
python scripts/launch_race.py --driver rule_based --laps 1

# Phase C (in progress)
python scripts/launch_race.py --driver optimal --laps 1  # crashes at ~480m
```

---

## Next Steps (Priority Order)

### 1. Debug & Fix OptimalLineDriver (BLOCKING for Phase C)
```python
# Option A: Add apex-seeking like rule_based
# In OptimalLineDriver._steer(), add curvature-based target offset:
sensors = state.track  # rangefinders
left_avg = (sensors[2] + sensors[3] + sensors[4]) / 3.0
right_avg = (sensors[14] + sensors[15] + sensors[16]) / 3.0
curvature = (left_avg - right_avg) / (left_avg + right_avg + eps)
target_tp += curvature * APEX_GAIN  # blend in curvature-based offset

# Option B: Check trajectory._idx() wrapping at finish line
# Add assertion: idx should never be >= len(buckets)

# Option C: Increase steering gains in OptimalLineDriver
# Try STEER_ANGLE_GAIN = 3.0, STEER_LINE_GAIN = 0.30
# Run one lap, check if 480m section is cleaner

# Option D: Telemetry-driven debug
# Run 1 lap with optimal driver --telemetry (if possible before crash)
# Plot distFromStart vs trackPos, steer, brake — see where it diverges
```

### 2. Phase C Complete (after fix)
- Run optimized OptimalLineDriver through 5 clean laps
- Add to ledger (expect −5 to −10 seconds based on early testing)
- Benchmark sector-by-sector vs Phase B using `--compare` flag

### 3. Phase D (Automated Tuning)
- Install `pip install cma` (CMA-ES optimizer)
- Parameterize trajectory (corner apex trackPos, speed scales, brake margins)
- Auto-tune 10–40 parameters over 100–200 laps (2–3 hours wall time)
- Save best params to `models/best_params.json`

### 4. Phase E (optional — RL fine-tuning)
- Record ≥5 of Phase D's fastest laps
- Train BC model on them
- RL warm-start with lap-time reward

---

## Key Files & Locations

| File | Purpose | Status |
|------|---------|--------|
| `laptime_ledger.csv` | Metrics ledger (append only) | ✅ Active |
| `scripts/benchmark.py` | Run driver → append ledger | ✅ Ready |
| `scripts/run_agent.py` | CLI launcher (load_driver) | ✅ Updated |
| `drivers/rule_based/driver.py` | Phase B baseline (ABS on) | ✅ Committed |
| `drivers/optimal/driver.py` | Phase C trajectory follower | ⚠️ Buggy, not committed |
| `torcs_env/track_map.py` | Track bucket data structure | ✅ Committed |
| `torcs_env/track_data/track_map.json` | Built map (uncommitted for now) | ⏸️ Rebuild after fix |

---

## How to Resume

1. **Run Phase B baseline** (verify it still works):
   ```bash
   conda run -n ai_env python scripts/launch_race.py --driver rule_based --laps 1
   ```

2. **Debug OptimalLineDriver crash**:
   - Check `drivers/optimal/driver.py` line ~90–110 (steer logic)
   - Add `print()` telemetry to _steer() to see angle/error/steer values at 400–500m
   - Test steering gain increases (3.0, 0.30) on a full lap

3. **Re-build track map** once fix is confirmed:
   ```bash
   conda run -n ai_env python scripts/build_track_map.py --telemetry data/rule_based_20260627_162255.csv
   ```

4. **Run Phase C**:
   ```bash
   conda run -n ai_env python scripts/launch_race.py --driver optimal --laps 3
   ```

---

## Technical Notes

- **TORCS timing**: Sim runs at ~280x real-time (8s wall ≈ 150s lap). ~50 Hz control loop.
- **distFromStart**: resets cleanly at 3608.4m (measured from telemetry). 5m buckets = 722 buckets.
- **ABS math**: front wheel spin ratio = `wheel_rad / (speed_ms / WHEEL_RADIUS)`. Lockup = ratio < 0.80. Reduction factor = `max(0, 1 − (threshold − ratio) / threshold)`.
- **Backward pass convergence**: usually 3–5 iterations (720 buckets, fast loop). Check for numerical stability if you add more complexity.

---

## Estimated Gains Remaining

- Phase C (OptimalLineDriver, once fixed): **−5 to −10 s** (late braking + racing line)
- Phase D (CMA-ES tuning): **−2 to −5 s** (fine-grained parameter sweep)
- Phase E (RL fine-tuning): **−0.5 to −2 s** (last few tenths if A–D plateau)

**Target**: < 130 s total lap time (from 151.7 s baseline).

---

## Questions for Next Agent

1. **Is the track map 84% corners realistic?** (curvature threshold = 0.05; may need lowering)
2. **Does OptimalLineDriver need to inherit from RuleBasedDriver for consistency?** (Currently BaseDriver; could mix rule_based steering + optimal speed control)
3. **Should Phase D params include BRAKE_DECEL_FACTOR tuning?** (Currently hardcoded; could be a sweep variable)

Cheers!
