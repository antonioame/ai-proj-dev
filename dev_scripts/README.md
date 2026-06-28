# Development Scripts

Temporary testing and debugging utilities. Not part of the main pipeline.

## Contents

- `check_data.py` — Inspect sensor statistics from recorded lap CSV
- `debug_bc_model.py` — Test BC model loading and inference on dummy sensor data
- `debug_model.py` — Compare BC vs RL model outputs
- `test_*.py` — Performance benchmarks (timing tests) and manual driver testing
  - `test_timing.py` — Client/action cycle timing
  - `test_env_timing.py` — Gym environment step timing
  - `test_collection_timing.py` — Experience collection timing
  - `test_sb3_timing.py` — Stable Baselines3 training timing
  - `test_all_models.py` — Benchmark all driver implementations
  - `test_bc_driver*.py` — Manual BC driver inference tests

## Usage

These are one-off utilities for development and debugging. They require:
- Active TORCS server (for timing tests)
- Specific CSV files or models (for data inspection)
- Manual modifications to file paths/parameters before running

**Not recommended for automated testing** — use `pytest tests/` instead.
