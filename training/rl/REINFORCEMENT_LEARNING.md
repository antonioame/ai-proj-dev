# Reinforcement Learning for the TORCS Driving Agent — Phase 3 Knowledge Base

> Suggested destination in the repo: `training/rl/REINFORCEMENT_LEARNING.md`
> (this replaces the Phase 3 placeholder `README.md` created during project scaffolding).

## 0. Why this document exists

This project currently has **Phase 1 (rule-based baseline)** and **Phase 2 (Behavioral Cloning)**
implemented, trained, and integrated. Both already produce a driving agent that completes the
Corkscrew lap (single lap, standing start, solo, no opponents, no crashes, minimal off-track
excursions — see project constraints in `CLAUDE.md`).

This document gives whoever implements **Phase 3 (Reinforcement Learning fine-tuning)** the
conceptual and practical knowledge needed to do it correctly, using the RL technique as taught in
the course (source: `10-Tecniche_TORCS.pdf`) plus the additional engineering knowledge required to
turn slides into a working, safe implementation.

**Two source types are mixed in this document and are labeled accordingly:**
- **[COURSE]** — content that comes directly from the course slide deck. Authoritative for the exam.
- **[ENGINEERING]** — general ML-engineering knowledge added to make the course technique
  implementable in this specific project. Not exam material — do not attribute it to the course.

---

## 1. Non-negotiable constraint

**Phase 3 must not regress Phase 1 or Phase 2.**

Concretely:
- Do **not** modify `torcs_env/client.py`, `torcs_env/sensors.py`, `torcs_env/actions.py`, the
  `rule_based` driver, or the trained BC model/weights. RL is **additive**: a new driver
  (e.g. `drivers/rl/`) that sits alongside the existing ones.
- The BC-trained model remains the **fallback/reference baseline**. Its current lap-completion
  behavior and lap time are the bar Phase 3 must meet or beat before it is ever promoted to
  "primary driver."
- Every RL checkpoint must be evaluated with the existing `scripts/evaluate.py` against the same
  metrics already used for the BC model (lap time, off-track fraction, damage/crashes). No
  checkpoint replaces the active driver unless it matches BC on safety (no crashes, comparable or
  better off-track fraction) **and** is not worse on lap time.
- If RL training destabilizes (diverges, produces a worse or unsafe policy), the implementation
  must be able to fall back to the BC driver with zero code changes elsewhere (driver selection via
  a flag/argument, as already done for `rule_based` vs `bc_model` in `run_agent.py`).

---

## 2. RL fundamentals **[COURSE]**

The agent-environment loop:
1. The agent observes state `s_t`.
2. It selects an action `a_t` following policy `π(s)`.
3. The environment returns reward `r_{t+1}` and next state `s_{t+1}`.
4. The agent updates its policy.

Objective — maximize the expected discounted return:

```
G_t = Σ_{k=0}^{∞} γ^k · r_{t+k+1}
```

- `γ` (discount factor) sets how much future reward matters relative to immediate reward.
  Typical range: 0.95–0.99.
- Exploration vs. exploitation trade-off: **ε-greedy** — with probability ε take a random action
  (explore), otherwise take the best known action (exploit).

### Q-Learning (tabular) **[COURSE]**

```
Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

- `α` = learning rate (typical: 0.1–0.5).
- `max_a' Q(s',a')` encodes the assumption of optimal future behavior — it is **not** "the action
  that deviates most from prior predictions" (a recurring point of confusion in study sessions).
- Not viable for this project as the main Phase 3 algorithm: TORCS' state space is continuous
  and high-dimensional (19 track rangefinders + speed + angle + trackPos, etc.); a Q-table
  explodes or requires destructive discretization that throws away precision the BC model already
  has. Q-Learning is only useful here as a conceptual stepping stone, not an implementation target.

### Why Deep RL instead **[COURSE]**

- Tabular Q-Learning doesn't generalize between similar states and can't handle continuous actions
  (precise `steer`, fine `accel`) natively.
- A neural network approximates `Q` (or the policy directly), generalizes across similar states,
  and handles continuous state/action spaces naturally.
- Trade-off: more powerful, but less stable and needs many more training episodes.

---

## 3. State and action space (recap, tied to existing code) **[COURSE + ENGINEERING]**

These match what `torcs_env/sensors.py` and `torcs_env/actions.py` already parse/serialize — no new
sensor plumbing should be needed.

**State (observations available):**
- `trackPos` (−1 left edge … +1 right edge), `angle` (car heading vs. track tangent)
- `track[19]` — rangefinder rays, 0°–180°
- `speedX`, `speedY`, `speedZ`
- `rpm`, `gear`, `wheelSpinVel[4]`
- optional: `focus[5]`, `opponents[36]` (not needed — solo race)

**Action (continuous, default space used by Phase 1/2):**
- `steer ∈ [−1, 1]`, `accel ∈ [0, 1]`, `brake ∈ [0, 1]`
- `gear`: keep using automatic gear as already implemented in Phase 1/2 — don't add manual gear
  control in Phase 3 unless BC/rule-based already exposed it, to keep the action space identical
  across drivers and make BC→RL weight transfer valid.

**[ENGINEERING]** Since Phase 3 will warm-start from the BC actor (Section 6), the RL policy
network's input/output layout **must exactly match** the BC model's input features and output
actions. Do not re-engineer features for RL; reuse the same feature extraction function BC already
uses.

---

## 4. Reward function

### 4.1 Baseline formula **[COURSE]**

```
r_t = v_x · cos(angle) − v_x · |sin(angle)| − v_x · |trackPos|
```

- `+ v_x · cos(angle)` — rewards longitudinal speed along the track direction.
- `− v_x · |sin(angle)|` — penalizes misalignment with the track tangent.
- `− v_x · |trackPos|` — penalizes distance from the track center.

Termination reward:
- Heavy penalty on leaving the track (e.g. `−100`).
- Optional bonus for completing the lap.

### 4.2 This is a starting point, not a final design **[ENGINEERING]**

Tony has confirmed the above should be used as a **baseline to refine empirically**, not adopted
verbatim. Concrete refinements the implementation should test and log (one change at a time, so the
effect of each is measurable):

1. **Progress reward.** Add a term proportional to distance traveled along the track centerline
   (not just instantaneous speed). This directly targets the project's actual success metric — a
   completed, fast lap — and is the standard mitigation the course slides give for reward hacking
   (see 4.3).
2. **Standing-still / spin-in-place guard.** The baseline reward can be gamed by low-speed
   maneuvers that avoid risk. Penalize `speedX` below a threshold sustained over N steps, or reset
   the episode if the car is stationary too long.
3. **Off-track termination severity.** Given the project's hard constraint ("no crashes, minimal
   off-track excursions"), the `−100` termination penalty should likely be tuned *higher* relative
   to per-step rewards than the slide default, so the policy strongly avoids track exits rather than
   trading a small chance of exiting for marginally higher speed.
4. **Consistency with BC warm start.** Since the actor starts from BC weights that already drive a
   full, safe lap, prefer reward shaping that is *potential-based* (adds no bias to the optimal
   policy) so early RL fine-tuning doesn't unlearn safe BC behavior in pursuit of quick reward
   gains. In practice: keep changes small, evaluate after every N training steps against the BC
   baseline, and stop/roll back a run that starts producing crashes the BC model didn't have.
5. **Normalization.** Keep all reward components in comparable magnitude ranges consistent with the
   input normalization already used for BC ([−1,1] or [0,1] per the course's own practical advice,
   Section 6).

Log the reward formula version used for every training run (in `CLAUDE.md` or a run config file) so
results are attributable to a specific reward design, not just "RL training."

### 4.3 Reward hacking — a specific risk with this formula **[COURSE]**

The slides explicitly warn: the agent can find shortcuts that maximize reward without driving well
— e.g., turning in place to accumulate reward with no risk of exiting the track. Mitigation given in
the course: penalize low speed, add a reward term for distance traveled. This is precisely refinement
#1 and #2 above — treat them as required, not optional, given this project's known failure mode.

---

## 5. Algorithm landscape **[COURSE]**

| Algorithm | Type | Continuous actions | Stability | Notes |
|---|---|---|---|---|
| Q-Learning (tabular) | value-based, discrete | No (needs discretization) | simple, comprehensible | good only as a first approach / basic lane-keeping |
| DDPG | actor-critic, off-policy | Native | sensitive to hyperparameters, needs tuning (replay buffer, target nets, OU noise) | best if you need maximum precision and can afford tuning |
| PPO | actor-critic, on-policy | Native | very stable, rarely diverges | simple with Stable-Baselines3, but on-policy → less sample-efficient, more episodes needed |
| SAC | actor-critic, off-policy | Native | off-policy like DDPG but much more stable; auto-tunes exploration via entropy | **recommended by the course slides for continuous control tasks like this one** |
| TD3 | actor-critic, off-policy | Native | improved DDPG (twin critics, delayed policy update), less overestimation | good alternative to DDPG |

Common Deep RL components (DDPG/SAC/TD3): **replay buffer** (breaks temporal correlation between
transitions), **target networks** with soft update `τ ≈ 0.001–0.005` (stabilize training against a
moving target). PPO instead uses a clipped surrogate objective:

```
L(θ) = min( r_t(θ)·A_t , clip(r_t(θ), 1−ε, 1+ε)·A_t ),  ε ≈ 0.2
```

which limits how much the policy can change in a single update.

---

## 6. Chosen approach for this project: SAC, warm-started from the BC actor

**Primary algorithm: SAC.** Rationale — off-policy sample efficiency (important given TORCS
episodes are slow to run, especially on the Windows headless server), native continuous action
support, and materially better training stability than DDPG, which matters because this project
cannot afford a divergent run silently corrupting a working baseline.

### 6.1 Hybrid BC → RL pipeline **[COURSE, this exact strategy]**

The slides explicitly recommend this hybrid for best effort/result ratio:
1. **Pre-training (already done — Phase 2):** the BC-trained network already provides a reasonable
   policy.
2. **Fine-tuning (Phase 3):** initialize the SAC actor's weights from the BC network, then continue
   training with RL to exceed the demonstrator's performance. Convergence is much faster than
   training SAC from scratch because it starts from an already-competent policy.

Implementation notes for this specific repo:
- Load the BC model's weights into the SAC actor network at initialization. If the BC network
  architecture differs from a standard SAC actor (e.g. different output activation), add an
  adapter layer rather than retraining feature extraction from scratch.
- Initialize the SAC critic separately (it has no BC equivalent) — a few thousand steps of critic-only
  warm-up before joint actor-critic updates begin can reduce early instability, since the actor
  starts "ahead" of an untrained critic.
- Keep the untouched BC model file as-is; save the RL-tuned model under a new name/path.

### 6.2 Hyperparameters **[ENGINEERING — not detailed for SAC in the course slides]**

The course slide deck gives a hyperparameter table only for DDPG (Slide 17). SAC shares most of the
same building blocks (replay buffer, target networks, actor-critic), so the DDPG table is a
reasonable anchor point, adapted with standard SAC defaults:

| Hyperparameter | Suggested value | Notes |
|---|---|---|
| Hidden layers / units | 2–3 layers, 64–256 units | matches course's general Deep RL guidance (Slide 30) |
| Learning rate | 3e-4 (Adam) | standard SAC default; course's DDPG table uses 1e-4 |
| Replay buffer size | 100K–1M transitions | course table suggests 50K+ for DDPG; SAC benefits from more |
| Batch size | 256 | |
| γ (discount) | 0.99 | course range is 0.95–0.99 |
| τ (soft update) | 0.005 | course DDPG table uses ~0.001; SAC commonly uses 0.005 |
| Entropy coefficient | auto-tuned (target entropy = −dim(action space)) | SAC-specific, no course equivalent |
| Activations | ReLU (hidden), tanh (action output) | matches course's practical advice (Slide 30) |

Treat these as a starting grid, not fixed values — tune based on the evaluation metrics in Section 1.

### 6.3 Practical environment integration **[ENGINEERING]**

- `torcs_env/client.py` is a raw UDP SCR client, not a Gym/Gymnasium environment. For
  Stable-Baselines3's SAC implementation, wrap it in a thin `gymnasium.Env` adapter
  (`training/rl/torcs_gym_env.py`) exposing `reset()` → obs and `step(action)` → (obs, reward, terminated,
  truncated, info). **Do not modify the underlying client** — wrap it.
- Use `stable_baselines3.SAC` (API identical to the `PPO` 3-line example already in the course
  slides: `SAC('MlpPolicy', env, ...).learn(total_timesteps=...)`).
- Disable TORCS rendering during training (headless, as already done for Phase 1/2) — the course
  notes this can speed up training substantially.

---

## 7. Known failure modes and mitigations **[COURSE]**

| Problem | Cause | Mitigation |
|---|---|---|
| Reward hacking | agent maximizes reward without driving well (e.g. spinning in place) | penalize low speed, reward distance traveled (Section 4.2) |
| Catastrophic forgetting | learning resets prior knowledge | experience replay (SAC has this natively); vary conditions during training |
| Local minima | agent settles for a safe-but-slow suboptimal policy | entropy bonus (SAC has this natively), aggressive resets on stagnation |
| Moving target | Q target shifts while the network trains, causing instability | soft-updated target networks (SAC has this natively) |

Given SAC already addresses replay/entropy/target-network issues natively, the main residual risk
for this project is **reward hacking against this project's specific constraints** (crashing,
off-track excursions) — hence Section 4.2's emphasis on that particular failure mode.

---

## 8. Practical advice carried over from the course **[COURSE]**

- Aggressive termination: off-track → reset episode.
- Checkpoint every ~50 episodes.
- Monitor mean reward **and** its standard deviation per episode, not just the mean.
- Normalize all inputs to [−1,1] or [0,1] (already done for BC — reuse it).
- If reward doesn't improve after ~100 episodes, change the reward or hyperparameters before
  assuming more training will fix it.
- Periodically render/inspect an episode to sanity-check what the policy is actually doing.
- Expect SAC/PPO-class algorithms to show decent results in 200K–500K steps; budget accordingly
  against the two-machine training setup (Mac M2 / MPS backend for training, Windows PC running the
  headless TORCS server).

---

## 9. Definition of done for Phase 3

- [ ] `training/rl/torcs_gym_env.py` — Gymnasium wrapper around the existing SCR client, no changes
      to `torcs_env/*`.
- [ ] `training/rl/train_sac.py` — training script, actor initialized from the Phase 2 BC weights,
      configurable reward version, checkpointing every ~50 episodes.
- [ ] `drivers/rl/driver.py` — new driver class implementing the same interface as
      `rule_based`/`bc_model` drivers, loadable via `run_agent.py --driver rl_model`.
- [ ] Reward function versioned and logged per training run.
- [ ] `scripts/evaluate.py` run against the RL driver, compared side-by-side against the existing BC
      driver's metrics (lap time, off-track fraction, crash count) — RL is only promoted to default
      if it matches or beats BC on safety and is not worse on lap time.
- [ ] `CLAUDE.md` updated: Phase 3 status, algorithm used, reward version, best lap time, and
      explicit confirmation that Phase 1/2 drivers remain unmodified and functional.

---

## References

- Course slide deck: `10-Tecniche_TORCS.pdf` (32 slides — environment/sensors, reward function,
  Q-Learning, Deep RL/DDPG/PPO, Imitation Learning, common pitfalls).
- `stable-baselines3.readthedocs.io` — SAC/PPO/DDPG/TD3 API reference.
- Course-cited external resources: `github.com/YurongYou/rlTORCS` (Deep RL examples on TORCS),
  `amslaurea.unibo.it` (Galletti, 2019 — DDPG vs PPO comparison thesis).
