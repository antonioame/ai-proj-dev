# Lap-Time Optimization Plan — Corkscrew, Single Lap, Standing Start, No Opponents

> **Audience:** the next agents, who **can launch TORCS, run laps on Corkscrew, read
> telemetry, and observe the race outcome**. This document is an actionable,
> measure-driven plan to minimise a single lap time in one specific scenario.
>
> **Hard scope:** Corkscrew track only · one lap · standing start · no opponents ·
> objective = **lap time (seconds), lower is better** · constraint = no crash, stay on track.
>
> Because the scenario is *fixed and singular*, **overfitting to Corkscrew is the goal,
> not a sin.** Every decision below trades generality for raw speed on this one track.

---

## 0. Strategic Thesis (read this first)

The current agent (`drivers/rule_based/driver.py`) is **reactive and myopic**: it decides
throttle/brake/steer purely from the 19 forward rangefinders (≤ ~200 m sight) and the
current `angle`/`trackPos`. It does **not know the track**. That is the correct design for a
*general* racer, and the wrong design for *the fastest possible lap on one known track*.

The winning move is to **convert the problem from "react to sensors" into "execute a known
optimal trajectory"**, exploiting two facts the current code ignores:

1. **The track is identical every run.** `distFromStart` (metres along the centreline) is a
   stable index. We can pre-compute, for every point on the track, the *target speed*,
   *target racing line* (`trackPos`), and *gear* — then just follow it. Offline trajectory
   optimisation beats online reaction every time, because it can look infinitely far ahead.
2. **There are no opponents and only one lap.** No overtaking, no defensive lines, no tyre/fuel
   management, no consistency-over-many-laps concern. We optimise a single deterministic run:
   the launch, one racing line, one braking profile. We may safely run on the absolute limit.

So the plan progresses from *instrument → exploit track knowledge → automate the tuning →
(optionally) learn the last few tenths*:

```
A. Instrument + map the track       (you cannot optimise what you cannot measure)
B. Quick wins on the reactive driver (launch + per-corner calibration)   ← tenths to ~1s
C. Track-position-indexed controller (racing line + speed profile)        ← the big win
D. Automated lap-time optimisation   (CMA-ES / Bayesian over parameters)  ← squeezes the limit
E. (Optional) Learn from best laps   (BC + targeted RL warm-start)        ← last few tenths
```

Phases A→D are deterministic, sample-efficient, and need **no GPU**. Phase E is optional and
only worth it once A–D plateau.

---

## 1. Definition of Done & Benchmark Protocol

**Before changing anything, establish a repeatable benchmark.** All claims of improvement
must be measured against it.

### 1.1 Metrics (record every run)
- **Lap time (s)** — primary objective. Standing start → cross finish line once.
- **Sector times (s)** — split the lap into N sectors (see §2) to localise where time is won/lost.
- **Top speed (km/h)** and **min corner speeds (km/h)** per corner.
- **Off-track excursions** — count + total `trackPos` overshoot beyond ±1.0.
- **Damage** — must remain 0. Any damage = invalid run.
- **Validity flag** — a run is valid only if: lap completed, damage == 0, no excursion that
  cuts the track.

### 1.2 Protocol
- Run **≥ 5 laps per configuration** (TORCS physics + control loop have small jitter). Report
  **best valid lap** (this is a time trial — best counts) **and median** (to detect a config
  that is fast-but-unreliable, which risks an invalid run).
- Always launch from the same standing start (`corkscrew_solo.xml`, standing start).
- Pin the control rate. Confirm the loop runs at TORCS' tick (~50 Hz) with no dropped packets;
  a stuttering client silently costs time. Log step count vs. wall time.
- Keep a results ledger: `results/laptime_ledger.csv` with columns
  `timestamp, config_id, git_sha, best_lap_s, median_lap_s, sector_times, top_speed, valid`.

### 1.3 Target
There is no published reference time, so the target is **monotonic improvement to a plateau**:
keep optimising until a phase yields < ~0.5% over several iterations, then move to the next
lever. The earlier exploration recorded the current driver completing a lap (e.g. baseline
`best_lap_s`); **first action is to measure the real current baseline** and write it to the ledger.

---

## 2. Phase A — Instrumentation & Track Mapping (foundation, do first)

You cannot optimise a track you have not measured. Build the map and the tooling.

### 2.1 Measure track length and finish line
- Drive one full recon lap with the existing rule-based driver, logging `distFromStart`,
  `distRaced`, `curLapTime`, `lastLapTime` every step.
- **Corkscrew length** = `distFromStart` value at the step *just before* it resets to ~0
  (this resolves Open Question #1 in `CLAUDE.md`). Record it as a constant `TRACK_LENGTH_M`.
- Confirm `distFromStart` is monotonic increasing within a lap and resets cleanly at the line.

### 2.2 Build the track map from telemetry
Create `torcs_env/track_map.py`. Drive 1–3 clean recon laps and derive, bucketed by
`distFromStart` (e.g. 5 m buckets):

| Field per bucket | How to derive |
|---|---|
| `s` (distFromStart, m) | bucket centre |
| `curvature` / `radius` | from change in heading: integrate `angle` + `speedY` over distance, **or** from the symmetric rangefinder asymmetry (`track[i]` vs `track[18-i]`), averaged over recon laps. Smooth heavily. |
| `corner_id` | segment the lap into **straights** and **numbered corners** by thresholding curvature. Label each. |
| `max_sight` | `track[9]` (centre rangefinder) — long on straights, short into corners. |

**Better, if available — read the TORCS track file directly.** TORCS tracks are defined by
segment files (`.../tracks/road/corkscrew/corkscrew.xml` or a `.trk`) listing each segment's
**type (straight/left/right), length, and radius**. If the next agents can read that file from
the TORCS install, compute the *exact* centreline geometry, corner radii, and corner positions
offline — far more accurate than telemetry inference. **Prefer this; fall back to telemetry
inference only if the track file is inaccessible.** Either way, the output is the same
`track_map` structure consumed by Phase C.

### 2.3 Sector definition
Partition `[0, TRACK_LENGTH_M)` into sectors at corner boundaries (e.g. one sector per
corner+following straight). Store sector boundaries in the track map. Sector timing = difference
in `curLapTime` as the car crosses each boundary `distFromStart`. Add this to the benchmark harness.

### 2.4 Benchmark harness
Create `scripts/benchmark.py`:
- Runs a chosen driver/config for K laps, computes the §1.1 metrics + sector times, appends to
  `results/laptime_ledger.csv`, and prints a per-sector comparison vs. the previous best config.
- Add a `--compare <config_id>` flag that prints a sector-by-sector delta table so you can see
  *exactly which corner* a change helped or hurt.

**Phase A exit gate:** track length known, `track_map.py` produces a usable corner/straight map,
sector timing works, baseline is in the ledger.

---

## 3. Phase B — Quick Wins on the Reactive Driver

These do not require the full trajectory controller and pay off immediately. Each is a
hypothesis to validate with `scripts/benchmark.py` (§1.2).

### 3.1 Launch / standing start (potentially the single biggest discrete gain)
The current "startup phase" is crude (`STARTUP_STEPS = 80`: 50% steer, full throttle,
speed-based gear). A standing start rewards a dedicated launch controller. Build `LaunchController`
(used for roughly the first ~3–5 s / until a speed threshold):

- **Traction-limited throttle, not blind full throttle.** Longitudinal grip peaks at a
  *small* slip ratio (~0.1–0.15), not at maximum wheelspin. Compute slip from `wheelSpinVel`
  (driven wheels) vs ground speed:
  `slip = (wheel_omega * WHEEL_RADIUS - v_ms) / max(v_ms, eps)`. Modulate throttle to hold
  slip near the optimum. Too much throttle off the line = wheelspin = slow launch.
- **Clutch control.** The `Action.clutch` field exists but is unused. From a standstill in
  gear, slipping the clutch keeps the engine in its power band instead of bogging/stalling.
  **Verify the SCR clutch convention empirically** (you can — run TORCS): determine whether
  `clutch=1` means engaged or disengaged, then implement a launch ramp (start partially
  disengaged, blend to fully engaged over ~0.5–1.5 s while holding target launch RPM).
- **Optimal launch RPM / gear.** Stay in gear 1, hold the engine near peak-power RPM during the
  clutch slip, then upshift at the optimal point (§3.3).
- **Steer straight** (Corkscrew start is presumably a straight — confirm from the map).

Iterate: sweep launch RPM target and clutch-release rate, measure time to a fixed
`distFromStart` (e.g. 100 m) and the resulting speed. Pick the fastest.

### 3.2 Threshold braking + precise braking points
- **Threshold braking:** maximum deceleration is at the friction limit; locking the wheels
  *reduces* grip and increases stopping distance. Detect incipient lockup via `wheelSpinVel`
  (a wheel rotating far slower than ground speed implies lockup) and ease off — a simple ABS.
  This lets you raise brake pressure (`BRAKE_MAX_*`) safely.
- **Calibrate `BRAKE_DECEL_FACTOR` empirically per the actual car.** Current value 255.0 is a
  guess for ~1.0 g. Do a controlled test: from a known speed on a straight, brake at full and
  measure the distance to a target speed → real deceleration → correct factor. Wrong factor =
  braking too early (slow) or too late (overshoot corner).
- **Trail braking:** keep some brake into corner entry and bleed it out as steering rises
  (the existing EBD logic is a start; tune `EBD_*` so the car rotates on entry without
  understanding wide).

### 3.3 Optimal gear-shift points
- Redline-only upshifting (`RPM_UPSHIFT = 9000`) is usually **not** optimal. The fastest
  upshift point is where wheel force in the next gear equals wheel force in the current gear
  (the crossover of `engine_torque(rpm) × gear_ratio` curves). This is often *below* redline.
- Empirically: on a straight, sweep the upshift RPM (e.g. 7000–9500 in steps), measure 0→top
  acceleration over a fixed distance, pick the best per gear.
- **Never shift mid-corner** (it upsets balance and TCS); the position-indexed controller
  (Phase C) can forbid shifts inside corner zones.

### 3.4 Corner-speed and racing-line tuning (manual, pre-Phase-C)
- Raise per-corner entry/apex speeds (loosen the `EDGE_*` caps and `TARGET_PHYSICS_SCALE`)
  until the car just starts to run wide, then back off slightly. Use sector deltas to confirm
  each corner improved without an excursion.
- **Prioritise corners that lead onto the longest straights** — exit speed there is multiplied
  over the whole following straight, so it dominates lap time. Identify these from the track map.

**Phase B exit gate:** launch, braking, gearing, and corner caps individually validated as
faster than baseline in the ledger, with damage == 0.

---

## 4. Phase C — Track-Position-Indexed Trajectory Controller (the big win)

This is where most of the lap time comes from. Build a controller that *follows a pre-computed
optimal trajectory* indexed by `distFromStart`, instead of reacting to sensors.

### 4.1 The trajectory data structure
Extend the track map into an optimal trajectory `track_map[s] → { target_speed, target_trackPos,
gear_hint }` where `s` = `distFromStart` bucket:

- **`target_trackPos` (the racing line):** out–in–out. Enter a corner from the outside, clip
  the apex (`trackPos` toward the inside edge), exit to the outside. Use **late apexes** for
  corners feeding long straights (sacrifice entry to maximise exit speed). Represent the line as
  a smooth `trackPos` curve over `s`.
- **`target_speed` (the speed profile):** the classic two-pass construction:
  1. **Cornering limit:** for each `s`, max speed from lateral grip `v ≤ sqrt(mu · g · R(s))`
     using the corner radius `R(s)` from the map (calibrate `mu` empirically per car/track).
  2. **Backward pass (braking):** walking backwards from each speed-limited point, cap the
     speed earlier so the car can decelerate in time at the friction limit → gives braking points.
  3. **Forward pass (acceleration):** walking forwards, cap by how fast the car can actually
     accelerate out (engine/traction limited) → smooths exits.
     The pointwise minimum of these is the speed profile. (This is the standard
     "friction-circle / forward-backward" lap-time-optimal speed profile.)
- **`gear_hint`:** from the target speed and the optimal shift map (§3.3).

### 4.2 The controller (`drivers/optimal/driver.py`, `OptimalLineDriver(BaseDriver)`)
At each step:
1. Read current `distFromStart`; look up the target trajectory at `s` and at a small
   speed-dependent **lookahead** `s + k·v` (so braking starts early enough).
2. **Speed control:** PI(D) on `(target_speed − current_speed)` → throttle when below,
   brake (threshold-limited, §3.2) when above. Use the *lookahead* target so it brakes before
   the corner, not in it.
3. **Line control:** steer toward `target_trackPos` (error = `trackPos − target_trackPos`),
   combined with `angle` correction. Reuse/retune the existing steering gains; the target is now
   the racing line, not track centre.
4. **Gear:** follow `gear_hint`, suppressed inside corner zones (§3.3).
5. **Launch:** delegate to `LaunchController` (§3.1) until past the launch zone.

### 4.3 Robustness
- **`distFromStart` glitches:** guard against the lap-reset discontinuity and any momentary
  jumps; fall back to the reactive rule-based controller if the lookup is invalid (defensive,
  so a bad index never crashes the car).
- **Keep the reactive driver as a safety fallback** if the car ends up far off the expected
  line (`|trackPos|` large or `angle` large) — recover, then resume the trajectory.

### 4.4 How to author the first trajectory
- Seed `target_trackPos` and `target_speed` from a clean Phase-B lap's telemetry (record the
  actual line/speed the tuned reactive driver took), then **sharpen it**: pull apexes tighter,
  push corner speeds up, straighten the line between corners. Re-benchmark after each edit using
  the sector-delta table.

**Phase C exit gate:** `OptimalLineDriver` beats the best Phase-B config on total lap time, with
clean sectors and damage == 0. This should be the largest single jump in the ledger.

---

## 5. Phase D — Automated Lap-Time Optimisation

Manual tuning plateaus. Since you can run a lap and read its time, treat lap time as a
**black-box objective** and let an optimiser tune the trajectory parameters.

### 5.1 Parameterise the trajectory
Expose a modest parameter vector (≈ 10–40 dims), e.g.:
- per-corner: apex `trackPos`, entry/apex/exit speed scale, braking-point offset, throttle-on
  point;
- global: `mu` (grip), launch RPM, clutch-release rate, upshift RPM per gear, lookahead gain,
  steering gains.

### 5.2 Optimiser
- **CMA-ES** (e.g. `pip install cma`) is the recommended default: derivative-free, robust to the
  simulator's noise, handles 10–40 continuous params well, and is far more sample-efficient than
  deep RL. **Bayesian optimisation** (e.g. `scikit-optimize`/`Ax`) is a fine alternative for
  fewer (< ~15) params and very expensive evaluations.
- **Objective:** `lap_time + LARGE_PENALTY · (off_track_overshoot) + HUGE · (damage > 0)`.
  Average 2–3 laps per evaluation to fight noise (or use median). Invalid/crashed run → return a
  large constant so the optimiser avoids that region.
- **Budget:** one lap ≈ a few minutes of wall time, so a few hundred evaluations is realistic
  overnight. Optimise **per sector / per corner where possible** to cut dimensionality (optimise
  the hardest sectors independently, holding others fixed).
- Log every evaluation to the ledger; checkpoint the best parameter vector to
  `models/best_params.json`.

**Phase D exit gate:** CMA-ES converges; best automated config is in the ledger and reproducible
from `models/best_params.json`.

---

## 6. Phase E — Learning-Based Refinement (optional ceiling-raiser)

Only pursue once A–D plateau and you want the last few tenths. This reuses the existing,
already-built Phase 2/3 infrastructure (`docs/PHASE2_BEHAVIORAL_CLONING.md`,
`docs/PHASE3_REINFORCEMENT_LEARNING.md`).

- **Behavioural cloning** on the *best* trajectory laps gives a smooth neural controller that
  generalises across the small state jitter better than a hand-coded lookup. Record ≥ 5 of the
  fastest clean laps (Phase C/D) → train `models/bc_v1.pth`.
- **Targeted RL fine-tuning** (Phase 3) with a **lap-time / progress reward**, warm-started from
  the BC policy, can discover micro-improvements (exact braking point, throttle modulation) a
  parameterised controller cannot express. Because the track is fixed, **let it overfit
  Corkscrew** — that is the objective here. Reward = forward progress − track-deviation −
  damage, with a large terminal bonus for completing the lap and a term proportional to
  `−lap_time`.
- Caveat: RL is the most expensive and least reliable lever; do not start it until Phases A–D
  have extracted their gains, or it will waste days chasing what CMA-ES finds in hours.

---

## 7. Deep Dive: The Technical Levers (reference)

| Lever | Why it matters here | Concrete action | Signal to use |
|---|---|---|---|
| **Launch** | Standing start; first seconds are pure dead time if mishandled | Clutch-slip launch + traction-limited throttle + optimal launch RPM | `wheelSpinVel`, `rpm`, `speedX`, `clutch` |
| **Exit speed onto straights** | Speed at corner exit is amplified along the whole straight — dominates lap time | Late apex + earliest possible full throttle on the corners before long straights | `track_map` (corner→straight), `trackPos`, `speedX` |
| **Braking point** | Braking too early = slow; too late = overshoot/excursion | Backward-pass braking profile + empirically calibrated decel + threshold braking (anti-lock) | `distFromStart`, `speedX`, `wheelSpinVel` |
| **Cornering speed** | Each corner has a grip-limited max | `v ≤ sqrt(mu·g·R)` from track map; push to the limit then back off | corner radius `R(s)`, `trackPos` |
| **Gear shift points** | Redline shifting is usually suboptimal | Shift at the gear-ratio force crossover; no mid-corner shifts | `rpm`, `gear`, `speedX` |
| **Racing line** | Shorter/faster path than track-centre | Out–in–out, late apex, indexed by `distFromStart` | `track_map`, `trackPos`, `angle` |
| **Control-loop health** | Dropped/late packets silently cost time | Verify ~50 Hz, no timeouts, no jitter | step count vs wall time |

---

## 8. New / Modified Files (concrete deliverables)

```
torcs_env/track_map.py            NEW  build + load track map; TRACK_LENGTH_M; corner/sector defs
drivers/launch/launch.py          NEW  LaunchController (clutch + traction-limited launch)
drivers/optimal/driver.py         NEW  OptimalLineDriver(BaseDriver) — position-indexed controller
drivers/optimal/trajectory.py     NEW  speed-profile (forward/backward pass) + racing-line builder
scripts/benchmark.py              NEW  K-lap benchmark, sector timing, ledger, --compare deltas
scripts/build_track_map.py        NEW  recon-lap → track_map artefact (or parse TORCS track file)
scripts/optimize_laptime.py       NEW  CMA-ES/BO loop over trajectory params → models/best_params.json
drivers/rule_based/driver.py      EDIT add LaunchController hook; expose corner caps as params
scripts/run_agent.py              EDIT register "optimal" driver; --config flag to load params
results/laptime_ledger.csv        NEW  the source of truth for "is it faster?"
models/best_params.json           NEW  best trajectory params from Phase D
models/bc_v1.pth                  (Phase E) BC checkpoint from best laps
```

Keep `OptimalLineDriver` behind the existing `BaseDriver` interface so it drops into
`scripts/run_agent.py` / `evaluate.py` / `benchmark.py` with no protocol changes.

---

## 9. Recommended Sequencing & Priority

1. **Phase A** (instrument + map) — *blocking*; everything downstream needs the map and the
   benchmark. ~highest leverage per hour because it makes all later work measurable.
2. **Phase B.1 launch** + **B.2 braking calibration** — biggest discrete quick wins.
3. **Phase C** — the structural win; expect the largest single ledger jump.
4. **Phase D** — automate to reach the limit of the parameterised controller.
5. **Phase E** — only if you want the last few tenths and A–D have plateaued.

At every step: **change one thing → run `scripts/benchmark.py` → read the sector deltas →
keep or revert.** The ledger is the arbiter, not intuition.

---

## 10. Risks & Pitfalls

- **Optimising for one fast lap that is unreliable.** A config that sets a great best lap but
  crashes 1-in-3 is a real risk in a one-shot scenario. Track *median* and *valid-rate*, not
  just best. The final delivered config must complete cleanly on demand.
- **Wrong `BRAKE_DECEL_FACTOR` / `mu`.** Mis-calibrated grip silently makes the whole speed
  profile too timid or too aggressive. Calibrate empirically before trusting the profile.
- **`distFromStart` discontinuities.** The lap reset and any sensor glitch must not index a bad
  trajectory bucket — guard and fall back to the reactive controller.
- **Mid-corner gear shifts** upset balance; forbid them in corner zones.
- **Clutch convention assumption.** Verify whether SCR `clutch=1` is engaged or disengaged in
  *this* TORCS build before building the launch ramp — do not assume.
- **Track-file vs telemetry map mismatch.** If you parse the TORCS track file, confirm its
  `distFromStart` origin/direction matches the SCR `distFromStart` the car reports (offset/sign).
- **Overfitting is intended here** — do *not* add generality "for robustness" that costs lap
  time. The deliverable is the fastest clean Corkscrew lap, nothing else.

---

## 11. Open Questions to Resolve Early (you can answer these by driving)

1. **`TRACK_LENGTH_M`** for Corkscrew (Phase A.1) — resolves `CLAUDE.md` Open Question #1.
2. **Is the TORCS Corkscrew track file readable** from the install? If yes, parse it for exact
   geometry instead of inferring from telemetry.
3. **SCR clutch convention** (engaged vs disengaged at 0/1) in this build.
4. **Empirical grip `mu` and real `BRAKE_DECEL_FACTOR`** from straight-line braking tests.
5. **Optimal upshift RPM per gear** from straight-line acceleration sweeps.
6. **Which corners feed the longest straights** (from the map) — these get priority for exit
   speed and late apexes.
```
