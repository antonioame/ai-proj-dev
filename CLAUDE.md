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

## Project Phases

### Phase 1 — DONE ✓ (current)
- Full SCR UDP client (`torcs_env/client.py`)
- Typed sensor parsing (`torcs_env/sensors.py`)
- Action serialisation (`torcs_env/actions.py`)
- Rule-based baseline driver (`drivers/rule_based/driver.py`)
- Telemetry recording script (`scripts/record_human.py`)
- Evaluation script (`scripts/evaluate.py`)
- Run script (`scripts/run_agent.py`)
- Race config XML (`torcs_env/race_config/corkscrew_solo.xml`)
- 37 unit tests — all passing

### Phase 2 — TODO
**Behavioral Cloning from recorded laps.**
- Prerequisites: ≥5 clean laps saved in `data/` (run `scripts/record_human.py`)
- Infrastructure **fully implemented** — dataset, model, train script, and `BCDriver` are all ready
  - `training/behavioral_cloning/dataset.py` — PyTorch Dataset from CSV
  - `training/behavioral_cloning/model.py`   — MLP policy (sensors → steer/accel/brake/gear)
  - `training/behavioral_cloning/train.py`   — training loop (MPS-aware, saves `.pth` checkpoint)
  - `drivers/bc/driver.py` — BCDriver with lazy checkpoint loading
- **Only missing:** ≥5 recorded laps + trained `models/bc_v1.pth`
- See `docs/PHASE2_BEHAVIORAL_CLONING.md` for the complete step-by-step guide.

### Phase 3 — TODO
**Reinforcement Learning fine-tuning (DDPG or PPO).**
- Wrap `TORCSClient` as a `gymnasium.Env` (`training/rl/gym_env.py` — not yet created)
- Use BC checkpoint to warm-start the policy
- Reward: forward progress minus track-deviation minus damage
- See `docs/PHASE3_REINFORCEMENT_LEARNING.md` for implementation plan + code templates.

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

## Repository Layout (quick reference)

```
torcs_env/        SCR protocol (sensors, actions, UDP client, race XML)
drivers/          Driving agents
  rule_based/     Phase 1 baseline (complete)
  bc/             Phase 2 — BCDriver ready, needs trained .pth
training/
  behavioral_cloning/   Dataset + MLP model + train script (all ready)
  rl/                   Placeholder for Phase 3 (README only)
scripts/          CLI entry points (run, record, evaluate, launch_race)
data/             Telemetry CSVs (git-ignored)
results/          Evaluation JSON files (git-ignored)
models/           Trained checkpoints (git-ignored, created by train.py)
tests/            37 unit tests
docs/             Detailed documentation (see below)
  ARCHITECTURE.md           System design, data flow, SCR protocol
  API_REFERENCE.md          All public classes and functions
  DEVELOPMENT_GUIDE.md      Setup, workflows, tuning, troubleshooting
  PHASE2_BEHAVIORAL_CLONING.md  Step-by-step BC implementation guide
  PHASE3_REINFORCEMENT_LEARNING.md  RL plan + code templates
old_project_material/  Legacy reference code (do not import)
```

---

## Open Questions / Handoff Notes

1. **Corkscrew track length** — needed for accurate lap detection via
   `distRaced`. Measure empirically after the first successful lap
   (read `distFromStart` as the car crosses the start/finish line).
2. **Rule-based tuning** — `STEER_ANGLE_GAIN`, `STEER_TRACK_GAIN`, and
   `SPEED_*` constants in `drivers/rule_based/driver.py` may need tuning
   for Corkscrew specifically. Run evaluate.py and iterate.
3. **SCR `scr_server` module** — confirm the module name matches what the
   installed SCR patch exposes. Some builds use `scr_server 0`, others `bt`.
   Adjust `corkscrew_solo.xml` if TORCS cannot find the driver.
4. **BC dataset quality** — rule-based recordings are useful for BC but
   may encode sub-optimal behaviour. Consider collecting human keyboard
   recordings once the basic loop is working.
