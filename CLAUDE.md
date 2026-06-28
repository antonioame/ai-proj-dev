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

## Project Status

### Current Drivers

**Phase 1: Rule-Based (WORKING)**
- Lap time: 149 seconds (stable, no crashes)
- Entry point: `scripts/run_agent.py --driver rule_based`
- No dependencies; pure algorithmic steering/speed control
- See `drivers/rule_based/driver.py` for tuning parameters

**Phase 2: Behavioral Cloning (REMOVED)**
- Attempted to train from recorded human data
- Issue: Driver crashed immediately (continuous steering, off-track on curves)
- Deleted: `training/behavioral_cloning/`, `drivers/bc/`, all `bc_*.pth` models
- Lesson: BC without proper normalization/dataset quality doesn't work

**Phase 3: Reinforcement Learning (REMOVED)**
- Attempted PPO/DDPG with BC warm-start
- Issue: Systematic observation space mismatch (env outputs 8 features, models expect 9)
- All RL models crashed or had zero steering
- Deleted: `training/rl/`, `drivers/rl/`, all model checkpoints

**Phase C: Optimal Line Driver (IN PROGRESS)**
- Status: Crashes but shows potential ("so fast")
- Issue: Too aggressive acceleration; needs speed tuning
- Entry point: `scripts/run_agent.py --driver optimal`
- See below for tuning guidance

---

## How to Run the Rule-Based Driver

```bash
# Windows: start TORCS server
torcs -r torcs_env/race_config/corkscrew_solo.xml

# Mac (or same machine): run the agent
TORCS_HOST=<windows-ip> python scripts/run_agent.py --driver rule_based
```

---

## How to Record Human / Baseline Data

```bash
# Windows: start TORCS server (same config)
torcs -r torcs_env/race_config/corkscrew_solo.xml

# Mac: record one lap
TORCS_HOST=<windows-ip> python scripts/record_human.py --driver rule_based
# Output: data/human_YYYYMMDD_HHMMSS.csv
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| UDP client only (no TORCS plugin) | SCR patch exposes a clean UDP interface; no C++ needed |
| `distRaced` reset detection for lap counting | `lastLapTime` only updates once per lap; distRaced is continuous |
| Proportional steering with `angle + trackPos` | Classic SCR baseline; works without a map |
| Curvature estimate from symmetric track sensor pairs | Simple, fast, no preprocessing required |
| Separate `clamp()` on Action | Keeps driver logic clean; clamping at the boundary not in every formula |
| PyTorch MPS device auto-detection | Mac M2 is the training machine; CUDA fallback for Linux/Windows |
| No hardcoded IP | `TORCS_HOST` env var — two-machine setup with no code changes |

---

## Repository Layout

```
torcs_env/        SCR protocol (sensors, actions, UDP client, race XML)
drivers/          Driving agents
  rule_based/     Phase 1 baseline (complete, working — 149s lap)
  optimal/        Phase C trajectory follower (in progress, needs speed tuning)
scripts/          CLI entry points (run_agent.py, record_human.py, evaluate.py)
tests/            Unit tests (all passing)
docs/             Documentation (see notes below)
  ARCHITECTURE.md, API_REFERENCE.md, DEVELOPMENT_GUIDE.md, etc.
  LAPTIME_OPTIMIZATION_PLAN.md  Performance ideas for Phase C
data/             Recorded telemetry CSVs (git-ignored)
results/          Evaluation JSON files (git-ignored)
models/           Trained checkpoints for optimal driver (if applicable)
dev_scripts/      Temporary test/debug utilities (not part of main pipeline)
old_project_material/  Legacy reference code (do not import)
```

**Deleted (broken):**
- `training/behavioral_cloning/` — BC training infrastructure
- `training/rl/` — RL training infrastructure  
- `drivers/bc/`, `drivers/rl/` — BC and RL drivers
- All model checkpoints (bc_*.pth, rl_*.zip)

---

## Next Priority: Tuning Optimal Line Driver

The **optimal driver** shows promise ("too fast, but has potential") but crashes at the first curve.
It needs conservative startup and smoother acceleration ramps.

**Key tuning parameters** in `drivers/optimal/driver.py`:
- `STARTUP_STEPS` (line 53): Duration of conservative startup phase. Increase from 80 to 150–200
- `SCAN_AHEAD_M` (line 38): Look-ahead distance for braking. Reduce from 250 to 150–200
- `BRAKE_MARGIN_M` (line 39): Brake onset buffer. Increase from 12 to 20–30
- `STEER_ANGLE_GAIN` (line 30): Steering sensitivity. Reduce from 2.0 to 1.0–1.5
- `STEER_SMOOTH_SPEED` (line 33): Speed threshold for steering smoothing. Increase from 40 to 60–80

**Testing:** After each change, run:
```bash
conda run -n ai_env python scripts/run_agent.py --driver optimal --laps 1
```

**Goal:** Complete a lap without crashing, aiming for lap time < 140 seconds.

---

## Deprecated / Removed

**Phase 2 (Behavioral Cloning)** — REMOVED  
Attempted to train on recorded human data. Failed due to:
- Continuous steering output (driver was overfitting to noise)
- Crashes on curves even with good training data
- Normalization/training loop issues made debugging difficult

**Phase 3 (Reinforcement Learning)** — REMOVED  
Attempted PPO/DDPG with BC warm-start. Failed due to:
- Observation space mismatch (8 vs 9 features) between training and inference
- All models exhibited zero steering or immediate crashes
- RL environment complexity + TORCS timeout issues made debugging slow

See commit `074c1ee` ("chore: remove all broken RL and BC models") for details.
