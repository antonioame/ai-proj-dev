# API Reference

Complete reference for all public classes and functions.

---

## `torcs_env.client` — TORCSClient

```python
class TORCSClient:
    lap: int          # current lap number (starts at 1)
    connected: bool   # True after successful handshake
```

### `__init__`

```python
TORCSClient(
    host: str = "localhost",
    port: int = 3001,
    sensor_angles: list[float] = DEFAULT_ANGLES,   # 19 angles
    timeout: float = 1.0,
    max_reconnect_attempts: int = 5,
)
```

Initialises a UDP socket but does **not** connect yet. Call `connect()` or use as a context manager.

`DEFAULT_ANGLES = [-45, -38, -30, -22, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15, 22, 30, 38, 45]`

---

### `connect() → None`

Sends the SCR init handshake and waits for `***identified***`. Raises `ConnectionError` if no response within `timeout` seconds (retried up to `max_reconnect_attempts` with exponential backoff).

---

### `receive() → SensorState | str`

Blocks until the next UDP packet arrives. Returns:
- `SensorState` — normal sensor frame
- `"RESTART"` — server sent `***restart***`
- `"SHUTDOWN"` — server sent `***shutdown***`

Also updates `self.lap` via `_update_lap(state.distRaced)`.

---

### `send(action: Action) → None`

Serialises `action` to the SCR control string and sends it via UDP. Calls `action.clamp()` internally before sending.

---

### `send_restart() → None`

Sends `***restart***` to request a new episode without closing the connection.

---

### `send_shutdown() → None`

Sends `***shutdown***` to terminate the simulation.

---

### `close() → None`

Closes the UDP socket. Safe to call multiple times.

---

### Context Manager

```python
with TORCSClient(host="192.168.1.100") as client:
    client.connect()
    state = client.receive()
    client.send(action)
```

`__exit__` calls `close()` automatically.

---

### Module-level Constants

```python
RESTART  = "RESTART"    # sentinel returned by receive()
SHUTDOWN = "SHUTDOWN"   # sentinel returned by receive()
```

---

## `torcs_env.sensors` — SensorState

```python
@dataclass
class SensorState:
    # Heading & lateral position
    angle: float          # heading error vs track axis (rad), ≈ [-π/2, π/2]
    trackPos: float       # lateral position, 0=centre, ±1=edge, >|1|=off-track

    # Speeds (km/h)
    speed: float          # alias for speedX (forward)
    speedY: float         # lateral speed
    speedZ: float         # vertical speed

    # Range sensors
    track: list[float]    # 19 distances (m), beams at DEFAULT_ANGLES
    opponents: list[float] # 36 opponent distances (m), every 10° around car

    # Engine
    rpm: float            # engine RPM
    gear: int             # current gear (-1=reverse, 0=neutral, 1-6)
    damage: float         # accumulated damage [0, 10000]

    # Distance & timing
    distRaced: float      # distance from start this lap (m)
    distFromStart: float  # distance from start line on the track (m)
    curLapTime: float     # elapsed time in current lap (s)
    lastLapTime: float    # time of completed last lap (s), 0 until first lap done

    # Wheel & chassis
    wheelSpinVel: list[float]  # 4 wheel angular velocities (rad/s) [FL, FR, RL, RR]
    z: float              # car height above ground (m)

    # Race info
    racePos: int          # race position (1=leading)
    fuel: float           # remaining fuel (litres)

    # Derived
    lap: int              # lap counter (set by TORCSClient, starts at 1)
    raw: str              # original sensor string (for debugging)
```

### `SensorState.from_string(sensor_str: str) → SensorState`

Parses a raw SCR sensor string into a typed `SensorState`. Uses regex
`r'\((\w+)\s+([^)]+)\)'` to extract `(key val ...)` tokens.

Unknown keys are silently ignored. Missing keys return field defaults (0 / empty list).

---

## `torcs_env.actions` — Action

```python
@dataclass
class Action:
    steer:  float = 0.0   # steering  [-1.0 (left), 1.0 (right)]
    accel:  float = 0.0   # throttle  [0.0, 1.0]
    brake:  float = 0.0   # brake     [0.0, 1.0]
    clutch: float = 0.0   # clutch    [0.0, 1.0]
    gear:   int   = 1     # gear      [-1 (reverse), 0 (neutral), 1-6]
    meta:   int   = 0     # 0=normal, 1=restart, 2=shutdown
```

### `Action.to_string() → str`

Returns the SCR control string, e.g.:

```
(accel 1.0000)(brake 0.0000)(clutch 0.0000)(gear 3)(meta 0)(steer -0.0200)
```

### `Action.clamp() → Action`

Returns a **new** `Action` with all values clamped to valid ranges. Does not mutate the original.

```
steer  → clamp(steer,  -1.0, 1.0)
accel  → clamp(accel,   0.0, 1.0)
brake  → clamp(brake,   0.0, 1.0)
clutch → clamp(clutch,  0.0, 1.0)
gear   → clamp(gear,   -1,   6)
meta   → unchanged
```

---

## `drivers.base_driver` — BaseDriver

```python
from abc import ABC, abstractmethod

class BaseDriver(ABC):
    @abstractmethod
    def step(self, state: SensorState) -> Action: ...

    def on_restart(self) -> None: ...    # called on RESTART sentinel
    def on_shutdown(self) -> None: ...   # called on SHUTDOWN sentinel
    def reset(self) -> None: ...         # called before a new episode
```

All non-abstract methods are no-ops by default. Override as needed.

---

## `rule_based_archived.driver` — RuleBasedDriver

Isolated driver, ~148 s baseline, not wired into `scripts/run_agent.py` — run it
standalone with `rule_based_archived/run_rule_based.py` (see `CLAUDE.md`).

```python
class RuleBasedDriver(BaseDriver):
    def step(self, state: SensorState) -> Action: ...
    def reset(self) -> None: ...         # resets PI integral and timers
```

All behaviour is controlled by module-level constants. Import and override them before instantiating the driver:

```python
from rule_based_archived import driver as d
d.STEER_ANGLE_GAIN = 2.5
d.MAX_SPEED = 180.0
drv = d.RuleBasedDriver()
```

### Tuning Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `STEER_ANGLE_GAIN` | 2.0 | Multiplier on `angle` (heading error) |
| `STEER_TRACK_GAIN` | 0.2 | Multiplier on lateral-position error |
| `STEER_LOCK` | 0.785398 | Max physical steer angle (rad, = 45°) |
| `CURVE_TRACKPOS_GAIN` | 0.30 | How far inward to aim on curves |
| `CURVE_TRACKPOS_MAX` | 0.28 | Maximum apex offset |
| `SMOOTH_SPEED_THRESH` | 42.0 | Below this speed (km/h): apply steer smoothing |
| `SMOOTH_ALPHA` | 0.7 | EMA coefficient for steer smoothing |
| `MAX_SPEED` | 200.0 | Absolute speed cap (km/h) |
| `BRAKE_DECEL_FACTOR` | 255.0 | Calibrated for ~1.0 g deceleration |
| `BRAKE_MARGIN` | 5.0 | Extra stopping distance headroom (m) |
| `TARGET_PHYSICS_SCALE` | 1.20 | Multiplier on physics-safe speed |
| `EDGE_HARD_SPEED` | 100.0 | Speed cap at trackPos > 0.88 |
| `EDGE_SOFT_SPEED` | 140.0 | Speed cap at trackPos > 0.75 |
| `FULL_THROTTLE_DIST` | 65.0 | Forward clear distance (m) for full throttle |
| `BRAKE_MAX_HIGH` | 0.65 | Max brake at speed > 140 km/h |
| `BRAKE_MAX_MED` | 0.78 | Max brake at 90–140 km/h |
| `BRAKE_MAX_LOW` | 0.90 | Max brake at speed < 90 km/h |
| `EBD_STEER_THRESH` | 0.08 | Steer magnitude to activate EBD |
| `EBD_GAIN` | 0.75 | Brake reduction multiplier when cornering |
| `EBD_FLOOR` | 0.40 | Minimum brake allowed by EBD |
| `THROTTLE_KP` | 0.40 | PI throttle proportional gain |
| `THROTTLE_KI` | 0.02 | PI throttle integral gain |
| `THROTTLE_MAX_INTEGRAL` | 1.0 | Integral anti-windup clamp |
| `TCS_STEER_THRESH` | 0.18 | |steer| to activate steering-based TCS |
| `TCS_GAIN_LOW_GEAR` | 1.45 | TCS cut multiplier in gears 1–2 |
| `TCS_GAIN_MID_GEAR` | 1.20 | TCS cut multiplier in gear 3 |
| `TCS_GAIN_HIGH_GEAR` | 0.70 | TCS cut multiplier in gears 4+ |
| `TCS_SLIP_THRESHOLD` | 1.25 | Rear wheel slip ratio to trigger cut |
| `RPM_UPSHIFT` | 9000 | RPM to upshift |
| `RPM_DOWNSHIFT_BY_GEAR` | {6:6800, 5:6300, 4:5800, 3:4300, 2:3500} | Downshift RPM per gear |
| `GEAR_SPEED_CAPS` | {1:15, 2:45, 3:75} | Max speed (km/h) per gear |
| `STARTUP_STEPS` | 80 | Steps with reduced steering (startup phase) |
| `STUCK_TIME_LIMIT` | 3.0 | Seconds stuck before reversing |
| `REVERSE_DURATION` | 2.0 | Seconds to reverse when stuck |
| `WHEEL_RADIUS` | 0.33 | Tyre radius estimate (m) for slip calculation |

---

## `bc_driver.driver` — BCDriver

**Driver principale, candidato alla consegna** (125.790 s, batte rule_based).

```python
class BCDriver(BaseDriver):
    def __init__(self): ...   # no args — loads both models synchronously at init
    def step(self, state: SensorState) -> Action: ...
    def on_restart(self) -> None: ...
    def reset(self) -> None: ...
```

Blend ibrido di due modelli `BCPolicy` (MLP, 26 input, hidden `[128, 64]`), caricati
da `bc_driver/models/` (path relativo a `bc_driver/driver.py`, non alla cwd):
- `bc_from_attempt1_v1.pth`/`.npz` — modello "rettilineo", allenato su dati generati
  dal driving-net in `bc_driver/bc_source_driver/attempt_model/`
- `bc_from_olddriver_v1.pth`/`.npz` — modello "curva"

Il blend è pesato da `state.track[9]` (distanza frontale): sopra `STRAIGHT_THRESHOLD`
(120 m) modello rettilineo puro, sotto `CORNER_THRESHOLD` (60 m) modello curva puro,
lineare in mezzo. Guadagni finali: `STEER_GAIN=1.8`, `ACCEL_GAIN=1.40`, `BRAKE_GAIN=0.80`.
Nei primi `STARTUP_STEPS` (80) passi, accelerazione piena e sterzo nullo per evitare
input fuori distribuzione a velocità ≈0.

Se uno dei quattro file modello manca, `__init__` solleva `FileNotFoundError` — non
c'è fallback silenzioso.

---

## `scripts.run_agent` — run()

```python
def run(
    laps: int = 1,
    host: str | None = None,   # overrides TORCS_HOST env var
    port: int | None = None,   # overrides TORCS_PORT env var
    save_telemetry: bool = False,
) -> dict:
    """
    Run the BC driver for `laps` laps. Returns metrics dict.
    Saves JSON to results/ and optional CSV to data/.
    """
```

### CLI

```bash
python scripts/run_agent.py \
    --laps 1 \
    --host 192.168.1.100 \
    --port 3001 \
    --telemetry
```

---

## `scripts.evaluate` — evaluate()

```python
def evaluate(
    laps: int = 1,
    host: str | None = None,
    port: int | None = None,
    output_path: str | None = None,   # default: results/eval_bc_{ts}.json
) -> dict:
    """
    Evaluate the BC driver and save structured metrics JSON.
    Returns the metrics dict.
    """
```

Output keys: `driver`, `evaluated_at`, `laps_requested`, `laps_completed`,
`lap_times_s`, `best_lap_s`, `avg_lap_s`, `max_speed_kmh`, `avg_speed_kmh`,
`off_track_pct`, `damage`, `total_steps`.

### CLI

```bash
python scripts/evaluate.py --laps 1
```

---

## `scripts.record_human` — record()

```python
def record(
    host: str | None = None,
    port: int | None = None,
) -> Path:
    """
    Shadow-record one complete lap (fixed neutral action) to data/human_{timestamp}.csv.
    Returns path to the saved CSV.
    """
```

Lap completion is detected by `state.lastLapTime > 0`.

### CLI

```bash
python scripts/record_human.py
```
