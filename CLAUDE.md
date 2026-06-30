# TORCS AI — Project Context for Claude Code Sessions

## Goal

Train an AI agent to complete a single lap of the **Corkscrew** circuit in TORCS
as fast as possible from a standing start, without crashing.

**Success metric:** Lap time (lower is better).  
**Constraints:** No crashes, minimal off-track excursions.

---

## Hardware Setup

| Machine | Role | Notes |
|---------|------|-------|
| Windows PC | TORCS headless server | Runs `torcs -r`, UDP port 3001 |
| MacBook Air M2 | Python client + training | PyTorch with MPS backend |

Both machines are on the same LAN. The Mac connects to the TORCS UDP server
via `TORCS_HOST=<windows-LAN-IP>` environment variable.

Key env vars:
```
TORCS_HOST   (default: localhost)
TORCS_PORT   (default: 3001)
```

---

## Car Livery Setup

The project includes a custom car livery (`livrea.png`) that is applied safely and reversibly.

**Install livery:**
```bash
conda run -n ai_env python scripts/setup_livery.py --install
```

**Check status:**
```bash
conda run -n ai_env python scripts/setup_livery.py --status
```

**Rollback to original (fully reversible):**
```bash
conda run -n ai_env python scripts/setup_livery.py --rollback
```

**How it works:**
- Converts `livrea.png` (PNG) → Radiance RGB format (TORCS native)
- Applies to `car1-stock1` car texture
- Automatic backup of original `car1-stock1.rgb` to `.rgb.backup`
- Can be rolled back to original without any loss

---

## Driver Status

### Phase 1: Rule-Based — DONE ✓ (stable baseline)
- **Lap time: ~148 s**, no crashes
- Entry point: `python scripts/run_agent.py --driver rule_based`
- Tuned with ABS, TCS, apex-seeking, PI throttle control
- See `drivers/rule_based/driver.py` for all constants

### Phase C: Optimal Line Driver — IN PROGRESS (crashes, needs testing)
- **Target: < 140 s** — trajectory-follower with late braking
- Entry point: `python scripts/run_agent.py --driver optimal`
- Requires `torcs_env/track_data/track_map.json` (already built from rule_based telemetry)
- **Known tuning as of this restructure:**
  - STARTUP_STEPS = 200 (conservative 4-second standing-start phase)
  - STEER_ANGLE_GAIN = 1.2 (was 1.6 — reduced to prevent twitching)
  - STEER_LINE_GAIN = 0.25 (was 0.40 — less aggressive line tracking)
  - STEER_SMOOTH_SPEED = 75 (apply EMA smoothing up to 75 km/h)
  - SCAN_AHEAD_M = 200 (was 300 — more focused look-ahead)
  - BRAKE_MARGIN_M = 40 (extra safety buffer on braking distance)
  - TARGET_LINE_SCALE = 0.50 (blend 50% toward racing line, 50% centre)
  - TCS added (prevents wheelspin on acceleration)
- **Rebuild map** if you record new telemetry:
  ```bash
  python scripts/build_track_map.py --telemetry data/<file>.csv
  ```

### Removed (broken, do not recreate without a plan)
- **Phase 2 Behavioral Cloning** — crashed immediately; continuous steering, no normalisation
- **Phase 3 Reinforcement Learning** — observation space mismatch; deleted

---

## How to Run

```bash
# 1. Start TORCS server (Windows)
torcs -r torcs_env/race_config/corkscrew_solo.xml

# 2. Run a driver (Mac or same machine)
conda run -n ai_env python scripts/run_agent.py --driver rule_based
conda run -n ai_env python scripts/run_agent.py --driver optimal

# 3. Record telemetry
conda run -n ai_env python scripts/record_agent.py --driver rule_based

# 4. Evaluate (saves JSON to results/)
conda run -n ai_env python scripts/evaluate.py --driver rule_based --laps 1
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| UDP client only (no TORCS plugin) | SCR patch exposes a clean UDP interface; no C++ needed |
| `distRaced` reset detection for lap counting | `lastLapTime` only updates once per lap; distRaced is continuous |
| `drivers/registry.py` for driver loading | Single source of truth — run_agent, record_agent, evaluate all use it |
| Physics-based speed target in rule_based | Braking distance formula, not lookup table — no step discontinuities |
| ABS on both drivers | Prevents lockup at high BRAKE_MAX values |
| TCS on both drivers | Prevents wheelspin on acceleration |
| Backward-pass trajectory | Propagates corner speed limits backwards to set braking points |
| `TARGET_LINE_SCALE = 0.50` | Blend racing line with centre to reduce off-track risk |

---

## Repository Layout

```
torcs_env/          SCR protocol (sensors, actions, UDP client, race XML)
  track_data/       track_map.json — prebuilt from rule_based telemetry
drivers/
  base_driver.py    Abstract interface
  registry.py       load_driver(name) — single loader used by all scripts
  rule_based/       Phase 1 baseline (~148 s, stable)
  optimal/          Phase C trajectory follower (in progress)
scripts/
  run_agent.py      Run any driver, optionally save telemetry + results JSON
  record_agent.py   Record a lap to data/recorded_<driver>_<ts>.csv
  evaluate.py       Evaluate and save structured results JSON
  build_track_map.py  Build track_map.json from a telemetry CSV
tests/              Unit tests
data/               Telemetry CSVs (git-ignored)
results/            Evaluation JSON files (git-ignored)
laptime_ledger.csv  Manual log of tuning experiments
```

---

## Lap Time Ledger

Record every benchmark run in `laptime_ledger.csv`:
```
timestamp,config_id,git_sha,best_lap_s,median_lap_s,top_speed_kmh,off_track_pct,damage,valid,notes
```

Current best: **148.4 s** (rule_based, ABS + higher brake pressure, commit ca54fea)

---

## Next Steps (ordered by priority)

1. **Test optimal driver** — does it complete a lap without crashing?
   ```bash
   conda run -n ai_env python scripts/run_agent.py --driver optimal --laps 1
   ```
2. **If still crashing** — reduce `STEER_ANGLE_GAIN` further (try 1.0) or increase `BRAKE_MARGIN_M` (try 60)
3. **If stable but slow** — increase `CORNER_SPEED_SCALE` in `drivers/optimal/trajectory.py` (try 1.1)
4. **Rebuild track map** with more laps of telemetry for better corner speed estimates
