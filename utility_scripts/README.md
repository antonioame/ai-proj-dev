# Development Scripts

Temporary testing and debugging utilities. Not part of the main pipeline.

## Contents

- `check_data.py` — Inspect sensor statistics from recorded lap CSV
- `test_timing.py` — Client/action cycle timing against a live TORCS server

## Usage

These are one-off utilities for development and debugging. They require:
- Active TORCS server (for timing tests)
- Specific CSV files or models (for data inspection)
- Manual modifications to file paths/parameters before running

**Not recommended for automated testing** — use `pytest tests/` instead.
