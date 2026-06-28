# Legacy RL Training Variants

Archived training scripts that were experimental or did not work.

## Why These Are Here

- **`train_rl_improved_reward.py`** — FAILED: Custom reward function variant that resulted in model crashes (~3.2 km). See `docs/ZERO_STEERING_SUMMARY.md` for analysis.
- **`train_rl_marathon.py`** — Experimental: Extended training session wrapper. Not validated; superseded by `train_rl_bc_warmstart.py`.
- **`train_rl_persistent.py`** — Experimental: Wrapper for persistent training across sessions. Not the active path.
- **`train_rl_final.py`** — Experimental: Earlier attempt at multi-session training. Superseded by more recent implementations.

## Active Training Script

Use **`train_rl_bc_warmstart.py`** for all RL training:

```bash
conda run -n ai_env python training/rl/train_rl_bc_warmstart.py --steps 100000
```

See `docs/PHASE3_REINFORCEMENT_LEARNING.md` for details.

## Reference

These files are preserved in git history for reference and reproducibility. Do not use them for new training runs.
