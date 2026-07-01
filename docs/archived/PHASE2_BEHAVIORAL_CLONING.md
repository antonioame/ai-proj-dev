# Phase 2: Behavioral Cloning

## Goal

Train an MLP policy on recorded lap telemetry so the BC driver can complete a clean lap faster than the rule-based baseline, or at minimum reproduce the baseline reliably.

**Status at Phase 1 handoff:**
- All training infrastructure is **ready** (dataset, model, train script)
- `BCDriver` is **ready** (lazy loads checkpoint at runtime)
- **Missing:** trained `models/bc_v1.pth` checkpoint
- **Required:** ≥ 5 clean lap recordings in `data/`

---

## Step-by-Step Implementation

### Step 1 — Record Training Data

Run the rule-based driver for at least 5 complete laps and save telemetry:

```bash
export TORCS_HOST=192.168.1.100

# Record one lap per invocation (lap detection: lastLapTime > 0)
for i in 1 2 3 4 5; do
    python scripts/record_human.py --driver rule_based
done
```

Each run produces `data/human_YYYYMMDD_HHMMSS.csv`. Inspect them:

```bash
wc -l data/human_*.csv          # should be several thousand rows each
head -2 data/human_*.csv        # verify schema
```

**Data quality checklist:**
- [ ] No rows with `damage > 0` (off-track / crash laps)
- [ ] `distRaced` increases monotonically within each file
- [ ] `speed` column has reasonable values (not all-zero)
- [ ] `trackPos` stays mostly in `[-0.8, 0.8]`

Discard any CSV where the car crashed or went significantly off-track.

---

### Step 2 — Train the Model

```bash
python -m training.behavioral_cloning.train \
    --data "data/human_*.csv" \
    --output models/bc_v1.pth \
    --epochs 50 \
    --batch-size 256
```

Expected console output:
```
Device: mps
Loaded 5 CSV files → 45231 rows
Train: 40707 | Val: 4524
Epoch  1/50 | train=0.1523 | val=0.1489
Epoch  5/50 | train=0.0934 | val=0.0921
...
Epoch 50/50 | train=0.0412 | val=0.0438
Checkpoint saved to models/bc_v1.pth
```

**Training converges when:**
- `val_loss` stops decreasing (within ≈ 5 epochs)
- `val_loss` ≈ `train_loss` (no large gap = no overfitting)

If val_loss diverges, reduce `--lr` to `5e-4` or increase `--epochs`.

---

### Step 3 — Evaluate the BC Driver

```bash
# Benchmark rule-based (baseline)
python scripts/evaluate.py --driver rule_based --laps 3

# Benchmark BC model
python scripts/evaluate.py --driver bc_model --laps 3
```

Compare `results/eval_*.json` files. Key metrics:

| Metric | Goal |
|--------|------|
| `best_lap_s` | BC ≤ rule_based |
| `off_track_pct` | BC < 5% |
| `damage` | BC = 0 |
| `laps_completed` | BC = laps_requested |

---

## Architecture Details

### Feature Engineering

The dataset uses 6 input features and 4 output targets:

**Inputs** (normalised to zero mean / unit std at training time):

| Feature | Column in CSV | Why |
|---------|--------------|-----|
| `speedX` | `speed` | Speed directly determines braking need |
| `trackPos` | `trackPos` | Lateral error drives corrective steer |
| `angle` | `angle` | Heading error drives steering |
| `rpm` | `rpm` | RPM drives gear selection |
| `gear` | `gear` | Current gear context for shifting |
| `damage` | — | Proxy for off-track severity (usually 0 in clean data) |

**Outputs:**

| Feature | Column | Type |
|---------|--------|------|
| `steer` | `steer` | Regression → Tanh output |
| `accel` | `accel` | Regression → Sigmoid output |
| `brake` | `brake` | Regression → Sigmoid output |
| `gear_out` | `gear_cmd` | Classification → 8-class softmax |

The gear output is treated as classification (not regression) because gear changes are discrete and MSE loss would predict averaged fractional gears.

### Model Architecture

```
Input (6) → Linear(6, 256) → LayerNorm → ReLU
          → Linear(256, 256) → LayerNorm → ReLU
          → Linear(256, 128) → LayerNorm → ReLU
               │
    ┌──────────┼──────────┬──────────┐
    ▼          ▼          ▼          ▼
Linear(128,1) Linear(128,1) Linear(128,1) Linear(128,8)
   Tanh         Sigmoid      Sigmoid       (gear logits)
   steer        accel        brake         gear class
```

### Loss Function

```python
# Per batch:
loss = (
    mse(steer_pred, steer_target) +
    mse(accel_pred, accel_target) +
    mse(brake_pred, brake_target) +
    cross_entropy(gear_logits, gear_target)
)
```

All four terms are summed with equal weight. If braking is poor, consider up-weighting brake: `2 × mse(brake_pred, brake_target)`.

---

## Checkpoint Loading in BCDriver

```python
ckpt = torch.load("models/bc_v1.pth", map_location="cpu")
model = MLPPolicy(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"])
model.load_state_dict(ckpt["model_state"])
model.eval()

sensor_mean = ckpt["sensor_mean"]  # np.ndarray shape (6,)
sensor_std  = ckpt["sensor_std"]   # np.ndarray shape (6,)
```

At inference time (inside `BCDriver.step`):
1. Extract the 6 features from `SensorState` in the same order as `SENSOR_COLS`
2. Normalise: `x = (x - sensor_mean) / (sensor_std + 1e-8)`
3. Convert to `torch.float32` tensor of shape `(1, 6)`
4. Call `model.predict(x)` → dict with `steer, accel, brake, gear`
5. Build `Action` and return

---

## Extending the BC Pipeline

### Using More Features

Edit `SENSOR_COLS` in `training/behavioral_cloning/dataset.py` and the feature extraction in `BCDriver._infer()` together. Keep them in sync.

Candidate additional features:
- `track_7`, `track_9`, `track_11` — left/centre/right lookahead
- `speedY` — lateral slip signal
- `wheelSpinVel_2`, `wheelSpinVel_3` — rear wheel spin (for slip)

### Larger / Different Architecture

Pass `--hidden-dims` to the train script (or modify `train.py`). The `hidden_dims` list is saved in the checkpoint, so `BCDriver` reconstructs the correct architecture automatically.

### Data Augmentation

Good augmentations for TORCS telemetry:
- **Mirror:** Flip `steer`, `trackPos`, `angle` sign simultaneously. Doubles dataset, helps symmetry.
- **Speed jitter:** Add small Gaussian noise to `speedX`.
- Do **not** augment `gear` or `rpm` (non-linear, easy to corrupt).

Implement augmentation in `TelemetryDataset.__getitem__`.

---

## Troubleshooting BC

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| BC driver crashes immediately | Steer output oscillating | Reduce `STEER_ANGLE_GAIN` equivalent; check normalisation stats match training |
| BC driver goes straight into wall | Model not loaded yet (fallback action) | Increase startup delay or add `_loaded.wait()` call |
| `val_loss` NaN | NaN in CSV data | Run `pd.read_csv(f).isna().sum()` on each file; drop bad rows |
| Gear always stuck at 1 | Gear loss dominated by other losses | Up-weight `cross_entropy` term by 2× |
| Model file not found | Wrong working directory | Run scripts from project root; path is relative to project root |
| Slow training | MPS/CUDA not detected | `python -c "import torch; print(torch.backends.mps.is_available())"` |
