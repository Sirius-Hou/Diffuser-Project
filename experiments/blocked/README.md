# Blocked corridor — B0, B1, B2, ours, oracle

| | |
|---|---|
| **Status** | Run on the real model (CPU, Lima VM): `smoke`, `probe` (12 full), `probe_partial` (8 partial). `main` not run yet |
| **Project step** | Main experiment of the proposal |
| **Run** | `bash experiments/blocked/run_all.sh smoke`, then `probe`, then `main` |

## Scenarios (`make_scenarios.py`)

For each seed we reset the env and sample the t = 0 plan as B0 does. The obstacle goes on a cell that **this plan** crosses at least `min_ahead_cells` cells after `t_appear`, so it always blocks the route the agent is taking. The start state and the t = 0 plan are stored (`scenarios.json`, `plans0.npz`) and every method starts from them.

- **full**: the whole cell. Kept only if A* finds a detour whose legs fit in the remaining steps.
- **partial**: `partial_frac` of the corridor width, on the side the plan uses. Straight corridor cells only.

Seeds with no usable cell are listed under `excluded`.

## Methods (`run_blocked.py --method ...`)

| Method | When it replans | What it does |
|---|---|---|
| B0 | never | follows the t = 0 plan |
| B1 | every `k_fixed` steps | best-of-N against the obstacle once it is known (one sample before that) |
| B2 | every `k` steps, if the rest of the plan hits the obstacle | best-of-N; if all N hit, takes the one that hits latest |
| ours | same trigger as B2 | best-of-N; if all N hit: A* around the blocked cell, junctions and turns become waypoints, one short plan per straight leg |
| oracle | at t = 0 and at `t_appear` | A* on the known map + speed-limited PD controller (gains untuned) |

The obstacle is a virtual region: the first step on which the ball overlaps it ends the episode as a failure. An episode stops at the first step within `success_radius` of the goal.

Settings shared by all methods: `n_samples`, `check_margin` (extra clearance in the predicted-collision check), and `replan_horizon`. With `"remaining"`, a plan to the goal is shortened so that it ends before the step limit; with `"full"` it is always 384 steps; with `"shrink"` it ends when the t = 0 plan would have ended.

Waypoint legs (ours): with `waypoint_mode: "corners"` every turn of the A* route is a waypoint, so each leg is straight (`r3_horizon` F3: bent legs cut through walls). A waypoint counts as reached when the ball is within `waypoint_radius` of the cell centre (0.5 = inside the cell); a leg is replanned if its plan runs out `leg_grace_steps` before that. Each leg is planned with `n_samples_leg` samples and a horizon of `steps_per_cell` per cell; with `leg_start_at_rest` the start velocity in the inpainted first state is set to zero (the ball still moves; it only changes the conditioning). Among the leg samples that are clear of the obstacle, the loop prefers one that stays on the A* route cells, then one with fewer points inside walls, then one that ends at rest.

## Outputs (`results/<config>/`)

`<method>/episodes/seedNNN.{json,npz,png}` per episode (the JSON has an `events` log: triggers, replans, A* calls, legs). `analyze_blocked.py` writes `episodes.csv`, `summary.json`, `summary.md`, `comparison.png`, and for B2/ours how many of the N samples were clear at the first trigger (risk R2).

## Configs

| Config | Purpose |
|---|---|
| `smoke.json` | 2 full + 1 partial scenario, N = 4: does the pipeline run |
| `probe.json` | 12 full scenarios, N = 16: R2 probe and the single-scenario detour check |
| `probe_partial.json` | 8 partial scenarios, N = 16 |
| `main.json` | 35 full + 15 partial scenarios |

`steps_per_cell` = 32 and `min_horizon` = 32 come from `r3_horizon` (F1, F2).

## Results (probe runs, CPU in the Lima VM)

All methods start from the same stored start state and t = 0 plan, so each row is paired by seed. n is small (12 + 8 scenarios): the Wilson intervals are wide, and the numbers say whether the method works, not how well. Full tables: `results/probe/summary.md`, `results/probe_partial/summary.md`.

### Full blockage (`probe`, 12 scenarios)

| Method | All 12 | Longer detour forced (9) | Same-length route exists (3) | Plans sampled / episode | Plan time / episode |
|---|---|---|---|---|---|
| B0 | 1/12 | 0/9 | 1/3 | 1 | 4 s |
| B1 | 3/12 | 1/9 | 2/3 | 79 | 159 s |
| B2 | 2/12 | 0/9 | 2/3 | 65 | 133 s |
| **ours** | **10/12** | **8/9** | 2/3 | 84 | 140 s |
| oracle | 12/12 | 9/9 | 3/3 | - | - |

- **The 11/23 stop-loss is met**: on 8 of the 9 scenarios that force a longer route, ours detours and reaches the goal while B0 and B2 fail on all 9 (e.g. seed 7: `results/probe/ours/episodes/seed007.png`, a 21-cell detour; B2 on the same seed hits the obstacle at step 345).
- **R2 did not happen**: at the first trigger on the forced-detour scenarios, 0 of 16 samples were clear of the obstacle (every scenario); with a same-length route, 12 of 16 were. So best-of-N alone cannot reroute around a full blockage, and A* is needed.
- "Same length" scenarios exist because the 384-step t = 0 plan often tours through cells that are not on a shortest route; blocking one of them leaves the shortest route open, and best-of-N is enough. They should be reported separately (as here) or filtered out of `main`.
- The time budget is tight: ours succeeds at a median of 670 of 800 steps on forced detours (the oracle: 597). Each leg's plan ends at rest on its waypoint, which probably slows the ball at every turn (not measured).
- Failures of ours:
  - seed 10, timeout: stuck on the leg (5,8)->(7,8). The model cut toward column 9 in 0/16 clean samples of that leg in a separate probe (`.vm/diag/leg_probe`, not in the repo).
  - seed 11, obstacle hit: the adopted plan cleared the obstacle by 0.27, but the ball tracked it with about 0.2 error. `check_margin` (0.05) is smaller than the tracking error; raising it conflicts with partial obstacles (next section).
- B1 (fixed interval, `k_fixed` = 100) succeeded once on a forced detour (seed 9). When all N samples hit, it keeps the one that hits latest (as B2 does) and tries again 100 steps later, so it reroutes only when a sample happens to take another corridor.

### Partial blockage (`probe_partial`, 8 scenarios, gap = 0.5)

| Method | All 8 | Detour exists (3) | No detour: must pass the gap (5) |
|---|---|---|---|
| B0 | 1/8 | 1/3 | 0/5 |
| B1 | 1/8 | 1/3 | 0/5 |
| B2 | 1/8 | 1/3 | 0/5 |
| **ours** | 3/8 | **3/3** | 0/5 |
| oracle | 7/8 | 3/3 | 4/5 |

- **The proposal's step 2 ("best-of-N finds a plan through the gap") does not hold at gap = 0.5**: at the first trigger, a median of 0 of 16 samples was clear (mean 0.4 when no detour exists). The model's plans run along the middle of the corridor, which the obstacle covers.
- When a detour exists: on seeds 6 and 7 all 16 samples hit, so ours treated the cell as blocked and took the 16- and 21-cell detours; on seed 1 a same-length route existed and best-of-N found it (12 of 16 clear). Ours: 3/3.
- When no detour exists, nobody but the oracle passes the gap. Ours falls back to the B2 behaviour (same outcome on all 5 seeds): on seeds 0 and 2 it adopted the only clear sample and the ball's tracking error carried it into the obstacle; on seeds 3 and 4 the adopted plans toured and stalled on a wall; on seed 5 no sample was ever clear and the ball ran into the obstacle.
- The oracle failure (seed 4) is an artifact: the oracle moves faster than the t = 0 plan, so it was already inside the cell when the obstacle appeared at t = 100 (`obstacle_collision_step` = `t_appear`).

### Not working (tried, kept out of the method)

- Interior inpainting (proposal option b, fix a via point mid-plan instead of chaining legs): 0 of 48 samples rerouted; the plans jump about 4.3 units to the via point and back (`.vm/diag/interior_probe`).
- Conditioning a leg on the ball's actual velocity: 18 of 48 leg samples clean, against 24 of 48 when the start is inpainted at rest (`.vm/diag/leg_velocity_probe`). Hence `leg_start_at_rest`.
