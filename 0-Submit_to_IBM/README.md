# TORCS AI — Corkscrew Driver

A behavioral-cloning driving agent that completes one lap of the Corkscrew
circuit in TORCS from a standing start. The agent connects to a running TORCS
server over UDP using the SCR protocol and drives the car using a hybrid of
two neural network models.

**Best recorded lap time:** 125.790 s, top speed 199.0 km/h.

## Contents

| Path | Description |
|---|---|
| `run_agent.py` | Main entry point: runs the driving agent against a TORCS server |
| `_DRIVER/` | Driving agent: `driver.py` (hybrid BC driver), `models/` (trained weights), `bc_source_driver/` (source driver used to generate training data) |
| `torcs_env/` | SCR protocol implementation: UDP client, sensor parsing, action encoding, race configuration |
| `scripts/` | Additional entry points: recording and evaluation |
| `livery/` | Car livery assets and installation scripts |
| `data/` | Recorded telemetry (CSV) used to train the driving models |

## How the driver works

`_DRIVER/driver.py` blends two models depending on track context:

- A model trained for straight-line driving.
- A model trained for cornering.

The blend weight is computed from the forward-facing track sensor, so the
agent smoothly transitions between the two models as it approaches a corner.

## Requirements

- Python 3.10+
- `torch`, `numpy`, `pandas`
- A running TORCS server with the SCR patch, listening for a UDP client

Install dependencies:

```bash
pip install torch numpy pandas
```

## Running the driver

1. Start the TORCS server with the Corkscrew race configuration:

   ```bash
   torcs -r torcs_env/race_config/corkscrew_solo.xml
   ```

2. From the project root, run the driving agent:

   ```bash
   python run_agent.py --laps 10
   ```

   The `--laps` flag defaults to 10 if omitted.

If the TORCS server runs on a different machine, set its address before
running the agent:

```bash
export TORCS_HOST=<server-ip>   # default: localhost
export TORCS_PORT=3001          # default: 3001
```

### Optional scripts

- Record telemetry while driving: `python scripts/record_agent.py --laps 1`
- Evaluate lap performance and save results to `results/`: `python scripts/evaluate.py --laps 1`

## Car livery

To (re)install the car livery, from the project root run:

```bash
python livery/setup_livery.py
```

See `livery/setup_livery.py --help` for additional options (install from a
custom PNG, restore the original livery, or roll back to the last backup).
