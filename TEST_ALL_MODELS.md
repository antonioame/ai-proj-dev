# Test All Models — Performance Evaluation

**Purpose:** Run each model one lap on Corkscrew track and record lap time + observations.

**Setup Before Running:**
1. Start TORCS server on Windows PC:
   ```bat
   cd U:\AI-Partition\torcs\torcs
   wtorcs.exe -r U:\AI-Partition\progetto_v2\ai_private_proj\torcs_env\race_config\corkscrew_solo.xml
   ```

2. Set environment variable on Mac/Linux client:
   ```bash
   export TORCS_HOST=<windows-pc-ip>  # e.g., 192.168.1.100
   ```

3. All commands run from project root in conda env:
   ```bash
   cd U:\AI-Partition\progetto_v2\ai_private_proj
   conda activate ai_env
   ```

---

## Test Commands

### 1. RULE-BASED BASELINE (Phase 1)
**Expected:** ~148 seconds, stable, no crashes

```bash
conda run -n ai_env python scripts/run_agent.py --driver rule_based --laps 1
```

**Result:**
- Lap time: 02:29:00 seconds
- Status: Completed
- Notes: The driver can be a lot better

---

### 2. BEHAVIORAL CLONING (Phase 2)
**Model:** `models/bc_v2.pth` (trained on human data)  
**Expected:** Should follow learned behavior from training data

```bash
conda run -n ai_env python scripts/run_agent.py --driver bc_model --laps 1
```

**Result:**
- Status: Crashed
- Notes: The driver is continuously steering, then it went off the track on the first curve.
it drives a shit.

---

### 3. RL BC WARM-START (Phase 3 — Primary)
**Model:** `models/rl_bc_warmstart/final.zip` (PPO, 50k steps with BC init)  
**Expected:** ~140-160 seconds (improvements possible)

```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_bc_warmstart --laps 1
```

**Result:**
- Status: Crashed
- Notes: Does not steer at all

---

### 4. RL DIRECT V1 (Alternative — Phase 3)
**Model:** `models/rl_direct_v1/final.zip` (PPO from scratch, no BC init)  
**Expected:** Varies; may be slower or unstable without warm-start

```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_direct_v1 --laps 1
```

**Result:**
- Error: Unexpected observation shape (8,) for Box environment, please use (9,) or (n_env, 9) for the observation shape.
---

### 5. RL FINAL V1 (Experimental — Phase 3)
**Model:** `models/rl_final_v1/final.zip`  
**Expected:** Unknown; experimental variant

```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_final_v1 --laps 1
```

**Result:**
- Error: Unexpected observation shape (8,) for Box environment, please use (9,) or (n_env, 9) for the observation shape.

---

### 6. RL MARATHON V1 (Experimental — Phase 3)
**Model:** `models/rl_marathon_v1/final.zip`  
**Expected:** Unknown; experimental variant

```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_marathon_v1 --laps 1
```

**Result:**
- Error: Unexpected observation shape (8,) for Box environment, please use (9,) or (n_env, 9) for the observation shape.

---

### 7. RL BC WARM-START V3 FIXED (Historical — Phase 3)
**Model:** `models/rl_bc_warmstart_v3_fixed/final.zip`  
**Expected:** May have issues; was being debugged

```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v3_fixed --laps 1
```

**Result:**
- Status: Timeout
- Notes: Went off track for a bit then stopped.

---

### 8. RL BC WARM-START V4 HOTFIX (Historical — Phase 3)
**Model:** `models/rl_bc_warmstart_v4_hotfix/final.zip`  
**Expected:** May have issues; was being debugged

```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v4_hotfix --laps 1
```

**Result:**
- Status: Crashed
- Notes: Gone directly into wall without steering.

---

### 9. OPTIMAL LINE DRIVER (Phase C — Experimental)
**Model:** None (trajectory-based algorithm)  
**Expected:** Untested; may be fast if track data is correct

```bash
conda run -n ai_env python scripts/run_agent.py --driver optimal --laps 1
```

**Result:**
- Status: Crashed
- Notes: So fast, but for this reason gone out at first curve

---

## Known Issues (Skip These)

❌ **RL IMPROVED REWARD V1** — **DO NOT RUN**  
Model: `models/rl_improved_reward_v1/final.zip`  
Reason: Documented crash at ~3.2 km; known failure from commit `5e6ed4d`

```bash
# Skip this — known to crash
# conda run -n ai_env python scripts/run_agent.py --driver rl_rl_improved_reward_v1 --laps 1
```

---

## Output to Expect

Each run will print live status like:
```
[hh:mm:ss] Step 50   | Speed: 125 km/h | Steering: -0.15 | Track pos: 0.0 m | Dist: 1234 m
[hh:mm:ss] Step 100  | Speed: 140 km/h | Steering:  0.02 | Track pos: 0.1 m | Dist: 2500 m
...
[hh:mm:ss] Lap completed in 142.3 seconds | Total distance: 5324 m
```

---

## Summary Template

After running all tests, fill this in:

| Model | Status | Lap Time | Notes |
|-------|--------|----------|-------|
| rule_based | | | |
| bc_model | | | |
| rl_bc_warmstart | | | |
| rl_rl_direct_v1 | | | |
| rl_rl_final_v1 | | | |
| rl_rl_marathon_v1 | | | |
| rl_rl_bc_warmstart_v3_fixed | | | |
| rl_rl_bc_warmstart_v4_hotfix | | | |
| optimal | | | |

---

## Next Steps

Once results are recorded:
1. Identify which model(s) perform best
2. Note any regressions or improvements
3. Decide if retraining is needed
4. Update docs/RUNNING_DRIVERS.md with new performance data
