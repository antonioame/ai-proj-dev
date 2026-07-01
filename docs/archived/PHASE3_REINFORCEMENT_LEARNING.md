# Phase 3: Reinforcement Learning

## Goal

Fine-tune or replace the BC policy using RL to push lap times below what behavioral cloning can achieve. The RL agent explores the state space and learns to exploit the physics simulation in ways not demonstrated in the training data.

**Status at Phase 2 handoff:** Placeholder directory `training/rl/` exists with a README. No code is implemented.

**Prerequisite:** A working BC checkpoint (`models/bc_v1.pth`) to warm-start the policy. RL from scratch on TORCS is extremely sample-inefficient.

---

## Planned Architecture

```
TORCSGymEnv (gymnasium.Env)
    ├── observation_space: Box(6,)     ← same 6 features as BC
    ├── action_space: Box(3,)          ← steer, accel, brake (continuous)
    │                                  (gear handled by rule-based logic)
    └── reward(state, action):
            r = v·cos(angle)            ← forward progress
              − |v|·|trackPos|          ← track deviation penalty
              − damage_delta × 100      ← crash penalty
              − (1 if off_track else 0) ← off-track penalty per step

RL Algorithm: DDPG (deterministic, continuous actions)
  or PPO (stochastic, easier to tune, slower)

Library: Stable-Baselines3
  pip install stable-baselines3
```

---

## Files to Create

```
training/rl/
├── gym_env.py        ← gymnasium.Env wrapping TORCSClient
├── reward.py         ← reward function(s) (separate for easy experimentation)
├── train_ddpg.py     ← DDPG training entry point
└── train_ppo.py      ← PPO training entry point (alternative)
```

---

## Step-by-Step Implementation

### Step 1 — Implement `TORCSGymEnv`

```python
# training/rl/gym_env.py
import gymnasium as gym
import numpy as np
from torcs_env.client import TORCSClient, RESTART
from torcs_env.sensors import SensorState
from torcs_env.actions import Action
from training.rl.reward import compute_reward

SENSOR_COLS = ["speedX", "trackPos", "angle", "rpm", "gear", "damage"]

class TORCSGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=3001, max_steps=10000):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        # steer ∈ [-1,1], accel ∈ [0,1], brake ∈ [0,1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self._host = host
        self._port = port
        self._max_steps = max_steps
        self._client: TORCSClient | None = None
        self._prev_state: SensorState | None = None
        self._steps = 0

    def _obs(self, state: SensorState) -> np.ndarray:
        return np.array(
            [state.speed, state.trackPos, state.angle,
             state.rpm, float(state.gear), state.damage],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._client is None:
            self._client = TORCSClient(self._host, self._port)
            self._client.connect()
        else:
            self._client.send_restart()
        result = self._client.receive()
        while result == RESTART:
            result = self._client.receive()
        self._prev_state = result
        self._steps = 0
        return self._obs(result), {}

    def step(self, action: np.ndarray):
        steer, accel, brake = float(action[0]), float(action[1]), float(action[2])
        # Rule-based gear (keeps RL action space smaller)
        gear = self._gear_from_rpm(self._prev_state)
        act = Action(steer=steer, accel=accel, brake=brake, gear=gear)
        self._client.send(act)

        result = self._client.receive()
        if result == RESTART or result == "SHUTDOWN":
            # Episode ended by server
            obs = self._obs(self._prev_state)
            return obs, -100.0, True, False, {}

        reward = compute_reward(self._prev_state, result)
        self._prev_state = result
        self._steps += 1
        terminated = self._client.lap > 1   # one lap completed
        truncated  = self._steps >= self._max_steps
        return self._obs(result), reward, terminated, truncated, {}

    def _gear_from_rpm(self, state: SensorState) -> int:
        rpm = state.rpm
        gear = state.gear if state.gear > 0 else 1
        if rpm > 9000 and gear < 6:
            return gear + 1
        if rpm < 5000 and gear > 1:
            return gear - 1
        return gear

    def close(self):
        if self._client:
            self._client.send_shutdown()
            self._client.close()
            self._client = None
```

### Step 2 — Implement `reward.py`

```python
# training/rl/reward.py
import math
from torcs_env.sensors import SensorState

def compute_reward(prev: SensorState, curr: SensorState) -> float:
    speed_ms = curr.speed / 3.6          # km/h → m/s
    progress = speed_ms * math.cos(curr.angle)    # forward velocity
    deviation = abs(speed_ms) * abs(curr.trackPos) # lateral drift penalty
    damage_delta = max(0.0, curr.damage - prev.damage)
    off_track = 1.0 if abs(curr.trackPos) > 1.0 else 0.0

    return (
          progress          # positive: moving forward
        - deviation         # negative: drifting sideways
        - damage_delta * 100.0  # large penalty for damage
        - off_track         # per-step off-track penalty
    )
```

### Step 3 — Warm-Start from BC Checkpoint

Stable-Baselines3's `MlpPolicy` uses a different architecture than `MLPPolicy`. The cleanest warm-start approach is to pre-train SB3's actor on the BC dataset rather than copying weights directly:

```python
# training/rl/train_ddpg.py
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from training.rl.gym_env import TORCSGymEnv

env = TORCSGymEnv(host="localhost", port=3001)

model = DDPG(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    device="mps",   # or "cuda" / "cpu"
)

# Optional: warm-start actor from BC checkpoint
# (requires custom weight mapping — see notes below)

model.learn(total_timesteps=500_000, log_interval=100)
model.save("models/ddpg_v1")
```

**Weight warm-start note:** SB3's actor network and `MLPPolicy` use different layer conventions. The easiest bridge is to train SB3's actor using supervised learning on the BC dataset for a few epochs before starting RL. Alternatively, use `model.policy.actor.set_parameters(...)` with careful key mapping.

### Step 4 — Evaluate

```bash
# Save the DDPG policy as a standalone driver (see below), then:
python scripts/evaluate.py --driver ddpg_model --laps 3
```

---

## Adding a DDPGDriver

Once `models/ddpg_v1.zip` is saved by SB3, create a driver that wraps it:

```python
# drivers/ddpg/driver.py
import numpy as np
import threading
from stable_baselines3 import DDPG
from drivers.base_driver import BaseDriver
from torcs_env.sensors import SensorState
from torcs_env.actions import Action

SENSOR_COLS = ["speedX", "trackPos", "angle", "rpm", "gear", "damage"]

class DDPGDriver(BaseDriver):
    def __init__(self, model_path: str = "models/ddpg_v1"):
        self._model_path = model_path
        self._model = None
        self._loaded = threading.Event()
        t = threading.Thread(target=self._load, daemon=True)
        t.start()

    def _load(self):
        self._model = DDPG.load(self._model_path, device="cpu")
        self._model.policy.set_training_mode(False)
        self._loaded.set()

    def step(self, state: SensorState) -> Action:
        if not self._loaded.is_set():
            return Action(accel=0.3, steer=0.0, brake=0.0, gear=1)
        obs = np.array(
            [state.speed, state.trackPos, state.angle,
             state.rpm, float(state.gear), state.damage],
            dtype=np.float32,
        )
        action, _ = self._model.predict(obs, deterministic=True)
        steer, accel, brake = float(action[0]), float(action[1]), float(action[2])
        return Action(steer=steer, accel=accel, brake=brake, gear=state.gear)
```

Register in `scripts/run_agent.py`:
```python
from drivers.ddpg.driver import DDPGDriver
DRIVERS["ddpg_model"] = DDPGDriver
```

---

## Reward Shaping Tips

| Behaviour to encourage | Add to reward |
|------------------------|--------------|
| Faster speeds | Increase coefficient on `progress` term |
| Staying on track | Increase `off_track` penalty (try 5.0) |
| Smooth steering | `− steering_delta² × 0.1` (penalise jerky inputs) |
| Completing the lap | `+500.0` terminal bonus when `lap > 1` |
| Avoiding damage | Increase damage penalty (try 500.0) |

Start with the simple reward and add terms only if the agent shows specific bad behaviours.

---

## Sample Efficiency Considerations

TORCS runs at ~50 Hz. One lap ≈ 12,000 steps ≈ 4 minutes of wall time.

| Approach | Steps to convergence | Wall time (estimate) |
|----------|---------------------|---------------------|
| DDPG from scratch | ~2M | ~110 hours |
| DDPG warm-started from BC | ~500K | ~27 hours |
| PPO warm-started from BC | ~1M | ~55 hours |

**Practical approach:** Train overnight (8–12 hours) and evaluate. If the agent is still crashing, add more BC pre-training steps or increase the damage penalty in the reward.

---

## Alternative: Vectorised Environments

TORCS does not natively support multiple simultaneous instances on the same UDP port. To run multiple parallel environments:

1. Run multiple TORCS instances on different ports: `3001, 3002, 3003, ...`
2. Pass different `port` values to each `TORCSGymEnv`
3. Use `VecEnv` from SB3 to wrap them

This requires multiple TORCS installations or configuration, but speeds up training proportionally to the number of instances.

---

## Open Questions for Phase 3

1. **Lap detection in gym:** `TORCSClient.lap` increments on `distRaced` reset. The gym `terminated` flag fires when `client.lap > 1`. Verify this triggers reliably at the finish line and not on recovery reversals.

2. **Episode resets:** `send_restart()` resets the car to standing start. Test that TORCS responds correctly and the next `receive()` gives a fresh state with `distRaced ≈ 0`.

3. **Gear handling in RL:** The action space above delegates gear to a rule-based logic to keep the RL action space smaller. An alternative is to include gear as a discrete action (multi-discrete space), which may allow the RL agent to find better shift points.

4. **Corkscrew track length:** Needed for normalising `distRaced` into a `[0, 1]` progress signal if used as part of the reward. Measure empirically from the first completed lap (`distFromStart` at the finish line).
