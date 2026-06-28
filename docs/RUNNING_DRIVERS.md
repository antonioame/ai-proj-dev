# Running Drivers — Quick Reference

## TL;DR Commit History

| Commit | What happened |
|--------|---------------|
| `56547e5` | Phase 3 done: PPO trained 50k steps with BC warm-start → `rl_bc_warmstart/final.zip` |
| `a8c0e68` | Attempted reward improvements: ×5 drift penalty, ×10 off-track penalty, lane-center bonus |
| `5e6ed4d` | **FAILED**: improved reward model still outputs zero steering, crashes at ~3.2 km. Root cause is NOT reward weights — it's the observation space (9 features), network size ([64,64]), or action pipeline |
| `10da2e2` | **Reset `reward.py` to last working state** (current) — see below before retraining |

---

## Step 1 — Start TORCS (Windows)

```bat
cd U:\AI-Partition\torcs\torcs
wtorcs.exe -r U:\AI-Partition\progetto_v2\ai_private_proj\torcs_env\race_config\corkscrew_solo.xml
```

---

## Step 2 — Run a Driver

All commands from the project root. Use the conda env:

```bash
conda run -n ai_env python scripts/run_agent.py --driver <DRIVER> [--laps 1] [--host <IP>] [--telemetry]
```

### Available Drivers

| `--driver` | Model used | Notes |
|------------|-----------|-------|
| `rule_based` | none (algorithmic) | **Best so far: 148.4 s** — use as baseline |
| `optimal` | none (trajectory-based) | Apex-seeking; use `--driver optimal` |
| `bc_model` | `models/bc_v1.pth` | Behavioral cloning v1 (5-feature) |
| `rl_model` | `models/ddpg_v1.zip` | DDPG (pure RL, early training) |
| `rl_ddpg` | `models/ddpg_v1.zip` | Same as above, explicit alias |
| `rl_ppo` | `models/ppo_v1.zip` | PPO (pure RL) |
| `rl_bc_warmstart` | `models/rl_bc_warmstart/final.zip` | **Best RL model** — PPO 50k steps, BC warm-start |
| `rl_rl_final_v1` | `models/rl_final_v1/final.zip` | RL final v1 |
| `rl_rl_marathon_v1` | `models/rl_marathon_v1/final.zip` | RL marathon v1 |
| `rl_rl_improved_reward_v1` | `models/rl_improved_reward_v1/final.zip` | **FAILED** — crashes at ~3.2 km |

> **Pattern for custom RL models:** `--driver rl_<folder>` resolves to `models/<folder>/final.zip` automatically.

### Examples

```bash
# Rule-based baseline (best performer)
conda run -n ai_env python scripts/run_agent.py --driver rule_based

# Best RL model (BC warm-start, 50k steps)
conda run -n ai_env python scripts/run_agent.py --driver rl_bc_warmstart

# Run on LAN Windows server, save telemetry
conda run -n ai_env python scripts/run_agent.py --driver rule_based --host 192.168.x.x --telemetry
```

---

## Retraining the RL Model

The reward function was **reset to the last working state** in commit `10da2e2`.
The current `training/rl/reward.py` weights are:

```
PROGRESS_WEIGHT  = 1.0   (forward velocity)
DEVIATION_WEIGHT = 1.0   (lateral drift penalty)
DAMAGE_WEIGHT    = 100.0
OFF_TRACK_PENALTY = 2.0
LAP_BONUS        = 500.0
```

### Why the improved reward failed

The zero-steering crash is **not** a reward problem. Known suspects:
1. Observation space too small (9 features — add more `track[]` sensors)
2. Network too shallow (`[64,64]` → try `[256,256]`)
3. Action normalization / steering clip wrong
4. TORCS session timeout (~1,600 steps max) starves training

### Retrain from BC warm-start (recommended)

```bash
conda run -n ai_env python training/rl/train_rl_bc_warmstart.py \
    --bc-model models/bc_v2.pth \
    --target-steps 100000 \
    --sessions 60 \
    --save-path models/rl_bc_warmstart_v2
```

Then run the new model:

```bash
conda run -n ai_env python scripts/run_agent.py --driver rl_rl_bc_warmstart_v2
```

### Retrain pure PPO (from scratch)

```bash
conda run -n ai_env python training/rl/train_ppo.py
```

---

## Evaluate Results

```bash
conda run -n ai_env python scripts/evaluate.py --driver rule_based
```

Results are saved as JSON in `results/`.
