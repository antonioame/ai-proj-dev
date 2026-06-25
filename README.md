# torcs-ai — AI Racing Agent for TORCS / Corkscrew

An AI agent that drives the Corkscrew circuit in TORCS as fast as possible,
built in three phases: rule-based baseline → behavioral cloning → RL fine-tuning.

## Architecture

```
Windows PC  ─── TORCS headless server (UDP :3001)
                        │
                   SCR protocol (UDP)
                        │
MacBook Air M2 ─── Python client + AI driver
```

---

## 1. Windows Setup (TORCS Server)

### 1a. Install TORCS 1.3.x

Download the TORCS 1.3.7 Windows installer from the official site and install to
`C:\torcs` (or any path without spaces).

### 1b. Install the SCR Patch

The SCR (Simulated Car Racing) patch adds a UDP server mode to TORCS.

1. Download the SCR patch for TORCS 1.3.x.
2. Copy `scr_server.dll` (and any companion files) into `C:\torcs\drivers\scr_server\`.
3. Verify TORCS can find the driver: start TORCS normally and check the driver list.

### 1c. Copy the Race Config

From this repository, copy `torcs_env/race_config/corkscrew_solo.xml` to any
convenient location on the Windows machine (e.g., `C:\torcs\race_config\`).

### 1d. Run TORCS Headlessly

```batch
cd C:\torcs
torcs.exe -r C:\torcs\race_config\corkscrew_solo.xml
```

TORCS will start without a window, load the Corkscrew track, and wait for a
UDP client on port 3001.

> **Firewall note:** Allow inbound UDP on port 3001 in Windows Firewall.
> Both machines must be on the same LAN (or route traffic appropriately).

---

## 2. Mac M2 Setup (Python Client)

### 2a. Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2b. Install PyTorch for Apple Silicon

```bash
# CPU build (also uses MPS automatically):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2c. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 2d. Confirm Tests Pass

```bash
pytest tests/ -v
# Expected: 37 passed
```

---

## 3. Running the Agent

### Quick Start

```bash
# Terminal 1 — Windows PC:
torcs.exe -r C:\torcs\race_config\corkscrew_solo.xml

# Terminal 2 — Mac (replace with your Windows LAN IP):
export TORCS_HOST=192.168.1.100
python scripts/run_agent.py --driver rule_based
```

The agent will connect, complete one lap, print the lap time, and exit.

### Options

```
python scripts/run_agent.py --driver rule_based --laps 3 --host 192.168.1.100 --port 3001
```

---

## 4. Recording Telemetry (Phase 2 Prep)

```bash
export TORCS_HOST=192.168.1.100
python scripts/record_human.py --driver rule_based
# Saves: data/human_YYYYMMDD_HHMMSS.csv
```

See `data/README.md` for the CSV schema and guidance on data quality.

---

## 5. Evaluation

```bash
python scripts/evaluate.py --driver rule_based --laps 1
# Saves: results/eval_rule_based_YYYYMMDD_HHMMSS.json
```

Reported metrics: lap time, max speed, avg speed, off-track %, damage.

---

## 6. Phase 2 — Behavioral Cloning

Once you have ≥5 clean laps recorded:

```bash
python -m training.behavioral_cloning.train \
    --data data/human_*.csv \
    --output models/bc_v1.pth \
    --epochs 50
```

Then implement `drivers/bc/driver.py` that loads `models/bc_v1.pth` and feeds
sensor observations through the MLP policy.

---

## 7. Project Structure

```
torcs_env/
  __init__.py
  client.py           # UDP client (SCR protocol)
  sensors.py          # Sensor string → SensorState dataclass
  actions.py          # Action dataclass → SCR control string
  race_config/
    corkscrew_solo.xml

drivers/
  base_driver.py      # Abstract BaseDriver
  rule_based/
    driver.py         # Phase 1: P-steering + PI speed control

training/
  behavioral_cloning/
    dataset.py        # PyTorch Dataset from CSV
    model.py          # MLP policy network
    train.py          # Training script (MPS-aware)
  rl/
    README.md         # Phase 3 plan

scripts/
  run_agent.py        # Run any driver
  record_human.py     # Record one lap to CSV
  evaluate.py         # Structured evaluation + JSON output

tests/                # 37 unit tests (pytest)
data/                 # Telemetry CSVs (git-ignored)
results/              # Evaluation JSONs (git-ignored)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ConnectionError: Could not connect to TORCS` | Check TORCS is running and `TORCS_HOST` / `TORCS_PORT` are correct; check Windows Firewall |
| TORCS exits immediately | Driver module name mismatch — edit `corkscrew_solo.xml` → `<attstr name="module" val="scr_server"/>` |
| Car immediately crashes | Tune `STEER_ANGLE_GAIN` and `STEER_TRACK_GAIN` in `drivers/rule_based/driver.py` |
| `TimeoutError` after a few seconds | TORCS lost the connection; restart both TORCS and the agent |
| MPS not available | Ensure PyTorch ≥ 2.1 and macOS ≥ 12.3; training falls back to CPU automatically |
