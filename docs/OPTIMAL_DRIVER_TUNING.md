# Tuning the Optimal Line Driver

## Status

**Current:** Crashes at first curve (too aggressive on acceleration)  
**Goal:** Complete a lap without crashing at lap time < 140s

## Quick Diagnosis

The optimal driver crashed with "so fast" note, meaning:
- ✅ Braking logic is sound (it knows when to brake)
- ✅ Trajectory following generally works
- ❌ Startup phase too aggressive OR steering response too slow
- ❌ Look-ahead window may be too long for sharp curves

## Tuning Strategy

### 1. Conservative Startup (First Change)

**File:** `drivers/optimal/driver.py` line 53  
**Current:** `STARTUP_STEPS = 80` (~1.6 seconds at 50 steps/sec)

During startup, steering is attenuated to 50% (line 86). If 80 steps isn't enough to stabilize:

```python
# Try this first
STARTUP_STEPS = 150  # or 200 for even more conservative

# This extends the period where steering is dampened:
steer = self._steer(state, 0.0) * 0.5  # 50% attenuation
```

**Test:** `conda run -n ai_env python scripts/run_agent.py --driver optimal --laps 1`

### 2. Reduce Steering Sensitivity (If Still Oscillating)

**File:** `drivers/optimal/driver.py` line 30

```python
# Current
STEER_ANGLE_GAIN = 2.0  # How much to react to car orientation error

# Try
STEER_ANGLE_GAIN = 1.0  # Halve the sensitivity
```

Lower values = smoother, less reactive. May feel sluggish if too low.

### 3. Adjust Brake Onset (If Crashing into Corners)

**File:** `drivers/optimal/driver.py` lines 38-39

```python
# Current
SCAN_AHEAD_M = 250.0     # Look-ahead distance (in meters)
BRAKE_MARGIN_M = 12.0    # Buffer before mathematical brake onset

# Conservative
SCAN_AHEAD_M = 150.0     # Shorter look-ahead = brake sooner
BRAKE_MARGIN_M = 25.0    # Larger buffer = brake earlier
```

Longer `BRAKE_MARGIN_M` = brakes sooner, smoother deceleration.

### 4. Smooth Steering at Low Speed (Fine-Tuning)

**File:** `drivers/optimal/driver.py` lines 33-34

```python
# Current
STEER_SMOOTH_SPEED = 40.0
STEER_SMOOTH_ALPHA = 0.35

# Try
STEER_SMOOTH_SPEED = 60.0  # Apply smoothing at higher speeds
STEER_SMOOTH_ALPHA = 0.50  # Increase damping ratio (0.0-1.0)
```

Higher `ALPHA` = more inertia in steering, smoother response.

## Recommended First Attempt

Start with this conservative configuration:

```python
# drivers/optimal/driver.py
STARTUP_STEPS = 150           # Extend startup phase
STEER_ANGLE_GAIN = 1.5        # Reduce steering sensitivity
STEER_SMOOTH_SPEED = 60.0     # Smooth at higher speeds
BRAKE_MARGIN_M = 25.0         # Brake earlier
SCAN_AHEAD_M = 180.0          # Moderate look-ahead
```

Then iterate:
1. Run lap → observe where it crashes
2. If on startup: increase `STARTUP_STEPS` more
3. If on curves: increase `BRAKE_MARGIN_M` or reduce `STEER_ANGLE_GAIN`
4. If oscillating: increase `STEER_SMOOTH_ALPHA`

## Expected Outcome

Once stable (no crashes), the optimal driver should target:
- **Lap time:** 120–140 seconds (vs rule-based 149s)
- **Behavior:** Late hard braking, efficient line following
- **Safety:** No off-track excursions

## Debug Tips

- The driver prints live status: `Speed: X km/h | Steer: Y | Track pos: Z m`
- Watch for "Track pos" value (should stay near 0; |value| > 1.0 = off-track)
- If stuck on startup: increase `STARTUP_STEPS` to 200+
- If crashes immediately: set `STEER_ANGLE_GAIN = 0.5` (very conservative)
