# Phase 3: Reinforcement Learning

This directory is a placeholder for the Phase 3 RL implementation.

## Planned Approach

**Algorithm:** DDPG or PPO via Stable-Baselines3  
**Environment:** `TORCSGymEnv` (gymnasium wrapper around `TORCSClient`)

### Reward Function (draft)

```
r = v * cos(angle)          # forward progress
  - |v| * |trackPos|        # penalise deviation from centre
  - damage_delta * 100      # penalise crashes
```

### Inputs to Phase 3

- Trained BC model (`models/bc_v1.pth`) — used to warm-start the RL policy  
- At least 5 complete human laps in `data/`  
- Tuned rule-based driver constants to use as a sanity-check baseline

### Files to Create

| File | Purpose |
|------|---------|
| `gym_env.py` | `gymnasium.Env` wrapping `TORCSClient` |
| `train_ddpg.py` | DDPG training entry point |
| `train_ppo.py` | PPO training entry point |
| `reward.py` | Reward function(s) — isolated for easy tuning |
