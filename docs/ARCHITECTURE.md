# Riferimento architettura

## Overview sistema

```
┌─────────────────────────────────────────────────────────────────┐
│  Windows PC                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TORCS 1.3.x + SCR patch                                │   │
│  │  - Simulates physics at ~50 Hz                          │   │
│  │  - Broadcasts sensor strings via UDP :3001              │   │
│  │  - Accepts control strings from UDP client              │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ UDP (SCR protocol)
                               │ LAN / localhost
┌──────────────────────────────┴──────────────────────────────────┐
│  Mac M2 / Linux                                                 │
│                                                                 │
│  torcs_env/client.py  ←──────────────────────────────────────  │
│      │  TORCSClient                                             │
│      │  - Handshake, send/receive loop                         │
│      │  - Lap counter via distRaced reset                       │
│      ▼                                                          │
│  torcs_env/sensors.py          torcs_env/actions.py            │
│      SensorState (dataclass)       Action (dataclass)          │
│      - 19-sensor rangefinder       - steer, accel, brake        │
│      - 36 opponent sensors         - gear, clutch, meta         │
│      - vehicle dynamics            - to_string() → SCR fmt     │
│      ▼                                 ▲                        │
│  drivers/base_driver.py                │                        │
│      BaseDriver (ABC)                  │                        │
│      .step(state) → action            │                        │
│           ▲                            │                        │
│     ┌─────┴──────┐                     │                        │
│     │            │                     │                        │
│  RuleBasedDriver  BCDriver ─── MLPPolicy.predict()             │
│  (Phase 1)       (Phase 2)     (loads .pth checkpoint)         │
│                                                                 │
│  scripts/                                                       │
│    run_agent.py    ← run a driver for N laps                   │
│    record_human.py ← record telemetry to CSV                   │
│    evaluate.py     ← structured metrics → JSON                 │
│    launch_race.py  ← Windows-only: start TORCS + agent         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Protocollo SCR

Il protocollo UDP SCR (Simulated Car Racing) è un ciclo request/response basato su testo.

### Handshake

```
Client → Server:  SCR(init -45 -38 -30 -22 -15 -10 -6 -3 -1 0 1 3 6 10 15 22 30 38 45)
Server → Client:  ***identified***
```

I 19 float nel messaggio init definiscono gli angoli del rangefinder (gradi), misurati dall'asse forward dell'auto. Negativo = sinistra, positivo = destra.

### Stringhe sensori

Ogni step simulazione (~20 ms / 50 Hz), il server invia una stringa sensori:

```
(angle 0.012)(curLapTime 12.345)(damage 0)(distFromStart 320.5)
(distRaced 320.5)(fuel 94.0)(gear 3)(lastLapTime 0)
(opponents 200 200 200 ... 200)   ← 36 valori
(racePos 1)(rpm 6500)(speedX 120.3)(speedY 0.1)(speedZ 0.0)
(track 12.3 14.1 18.0 ...)        ← 19 valori
(trackPos 0.02)(wheelSpinVel 25.1 25.2 25.0 25.1)(z 0.34)
```

### Stringhe di controllo

Il client deve rispondere entro la finestra di step con:

```
(accel 1.0)(brake 0.0)(clutch 0.0)(gear 3)(meta 0)(steer -0.0200)
```

### Sentinelle

| String | Direzione | Significato |
|--------|-----------|-------------|
| `***identified***` | Server→Client | Handshake accettato |
| `***restart***` | Client→Server | Richiedi riavvio episodio |
| `***shutdown***` | Client→Server | Termina simulazione |

---

## Module Dependency Graph

```
scripts/run_agent.py
  └─ drivers/rule_based/driver.py  (RuleBasedDriver)
  └─ drivers/bc/driver.py          (BCDriver)
       └─ training/behavioral_cloning/model.py  (MLPPolicy)
  └─ torcs_env/client.py           (TORCSClient)
       └─ torcs_env/sensors.py     (SensorState)
       └─ torcs_env/actions.py     (Action)

training/behavioral_cloning/train.py
  └─ training/behavioral_cloning/dataset.py  (TelemetryDataset)
  └─ training/behavioral_cloning/model.py    (MLPPolicy)
```

No circular imports. Each layer only depends downward.

---

## Data Flow: Inference Loop

```
1. TORCSClient.receive()
       │ raw UDP bytes → strip null → decode UTF-8
       ▼
2. SensorState.from_string(raw_str)
       │ regex tokenise "(key val...)" → typed dataclass fields
       ▼
3. BaseDriver.step(state) → Action
       │ RuleBasedDriver: P-control steering + PI throttle + physics braking
       │ BCDriver: normalise features → MLPPolicy.predict() → Action
       ▼
4. Action.clamp() → clamped Action
       ▼
5. TORCSClient.send(action)
       │ action.to_string() → UTF-8 bytes → UDP socket
       ▼
6. (repeat at ~50 Hz)
```

---

## Lap Counter Design

TORCS resets `distRaced` to ~0 each time the car crosses the start/finish line. The client detects this as a drop > 100 m and increments `client.lap`:

```python
if prev_dist - dist_raced > 100:
    self.lap += 1
```

`lastLapTime` only updates once per lap (after crossing the line), making it unsuitable for step-by-step detection. `distRaced` is updated every step.

---

## Rule-Based Driver: Control Architecture

```
SensorState
    │
    ├─ _compute_steering(state)
    │      angle × STEER_ANGLE_GAIN
    │    − (trackPos − target_tp) × STEER_TRACK_GAIN
    │      normalised by STEER_LOCK (45°)
    │      target_tp from curvature estimate (sensor asymmetry)
    │
    ├─ _target_speed(state)
    │      forward_dist = track[9] (centre sensor)
    │      physics_safe = sqrt(forward_dist × BRAKE_DECEL_FACTOR)
    │      target = min(MAX_SPEED, physics_safe × TARGET_PHYSICS_SCALE)
    │      edge limiters: hard 100 km/h at trackPos>0.88
    │
    ├─ _compute_throttle_brake(state, steer, target_speed)
    │      Priority 1: Emergency brake (wall within stopping dist)
    │      Priority 2: Coast (speed > target)
    │      Priority 3: Full throttle (forward dist ≥ 65 m)
    │      Priority 4: PI control (KP=0.40, KI=0.02)
    │
    ├─ _traction_control(state, steer, accel) → accel
    │      Steering-based cut: |steer| > TCS_STEER_THRESH
    │      Slip-based cut: rear_spin / expected > 1.25
    │
    ├─ _compute_gear(state) → gear
    │      Upshift: RPM > 9000
    │      Downshift: gear-specific RPM thresholds
    │      Speed caps: gear 1 ≤15 km/h, gear 2 ≤45 km/h
    │
    └─ _stuck_recovery(state) → Action | None
           Detects stuck (trackPos>0.9 OR speed<5 for >3 s)
           Returns reverse action for REVERSE_DURATION (2 s)
```

---

## Behavioral Cloning Driver: Inference Pipeline

```
SensorState
    │
    │  Features (in order):
    │  ["speedX", "trackPos", "angle", "rpm", "gear", "damage"]
    │
    ├─ Normalise: (x − mean) / std   (stats saved in .pth checkpoint)
    │
    ├─ MLPPolicy.predict(x)
    │      Backbone: Linear→LayerNorm→ReLU × 3 layers [256, 256, 128]
    │      Steer head:  Tanh → float in [-1, 1]
    │      Accel head:  Sigmoid → float in [0, 1]
    │      Brake head:  Sigmoid → float in [0, 1]
    │      Gear head:   argmax(8 logits) − 1 → int in [-1, 6]
    │
    └─ Action.clamp() → send
```

---

## Checkpoint Format (`models/bc_v1.pth`)

The checkpoint is a plain Python dict saved with `torch.save`:

```python
{
    "model_state": OrderedDict,      # nn.Module.state_dict()
    "input_dim": int,                # 6
    "output_dim": int,               # 4 (steer, accel, brake, gear_out)
    "sensor_mean": np.ndarray,       # shape (6,) — feature means
    "sensor_std": np.ndarray,        # shape (6,) — feature stds
    "hidden_dims": list[int],        # [256, 256, 128]
    "history": list[dict],           # per-epoch train/val losses
}
```

BCDriver loads this and reconstructs `MLPPolicy(input_dim, hidden_dims)`.

---

## CSV Telemetry Schema

Files in `data/` follow this schema (written by `record_human.py` and `run_agent.py --telemetry`):

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Unix timestamp |
| `angle` | float | Heading error vs track axis (rad) |
| `speed` | float | Forward speed (km/h) |
| `trackPos` | float | Lateral position [−1, 1], 0 = centre |
| `track_0`…`track_18` | float | Rangefinder distances (m), sensors at −45°…+45° |
| `rpm` | float | Engine RPM |
| `gear` | int | Current gear (−1=reverse, 0=neutral, 1–6) |
| `distRaced` | float | Distance driven this lap (m) |
| `curLapTime` | float | Elapsed time in current lap (s) |
| `steer` | float | Steering command sent [−1, 1] |
| `accel` | float | Throttle command sent [0, 1] |
| `brake` | float | Brake command sent [0, 1] |
| `gear_cmd` | int | Gear command sent |

The `training/behavioral_cloning/dataset.py` uses a subset:
- **Inputs:** `speedX, trackPos, angle, rpm, gear, damage`
- **Outputs:** `steer, accel, brake, gear_out`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCS_HOST` | `localhost` | IP of the machine running TORCS |
| `TORCS_PORT` | `3001` | UDP port of the SCR server |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| UDP-only (no TORCS plugin) | SCR patch exposes a clean UDP interface; no C++ compilation needed |
| `distRaced` reset for lap detection | `lastLapTime` updates once per lap; `distRaced` is continuous |
| Apex-seeking target trackPos | Cuts corners slightly; no map needed — uses live sensor asymmetry |
| Physics-based braking distance | No lookup tables; adapts continuously to actual speed and lookahead |
| Lazy model loading in BCDriver | Avoids SCR handshake timeout (< 2 s budget) during model load |
| PyTorch MPS auto-detection | Mac M2 is primary training machine; falls back to CUDA then CPU |
| 19 rangefinder beams at fixed angles | Covers ±45° with high resolution near centre; good curvature signal |
| 4-head MLP (continuous + discrete) | Steer/accel/brake are regression; gear is classification (8 classes) |
