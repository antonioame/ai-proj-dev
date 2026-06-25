# Data Directory — Human Telemetry

This directory stores recorded telemetry CSV files used for Phase 2 Behavioral Cloning.

## CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | Unix timestamp (seconds) |
| angle | float | Car angle vs track axis (rad) |
| speed | float | Longitudinal speed (km/h) |
| speedY | float | Lateral speed (km/h) |
| speedZ | float | Vertical speed (km/h) |
| trackPos | float | Track position (0=centre, ±1=edge) |
| track_0 … track_18 | float | Rangefinder readings (metres, max 200) |
| rpm | float | Engine RPM |
| gear | int | Current gear (-1=reverse, 0=N, 1–6) |
| distRaced | float | Cumulative distance raced (m) |
| curLapTime | float | Time in current lap (s) |
| steer | float | Steering command [-1, 1] |
| accel | float | Throttle command [0, 1] |
| brake | float | Brake command [0, 1] |
| gear_cmd | int | Gear command sent to TORCS |

## How to Record a Lap

### Option A: Record rule-based driver (immediate, no human required)

```bash
# Mac / Linux (with TORCS running on Windows at TORCS_HOST)
TORCS_HOST=192.168.1.X python scripts/record_human.py --driver rule_based
```

### Option B: Record while the rule-based driver also sends actions
Since SCR does not support a true observer mode, `record_human.py` runs the
selected driver and records its sensor+action data simultaneously.

## File Naming

Files are auto-named `human_YYYYMMDD_HHMMSS.csv`.  
Keep at least **5 complete laps** before starting Phase 2 training.

## What Happens to Large Files

`*.csv` files are excluded from git (see `.gitignore`).  
If you need to share recordings between machines, use a shared network folder
or upload to a cloud bucket and reference the path in your training command.
