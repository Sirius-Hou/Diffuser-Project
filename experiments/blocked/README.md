# Blocked corridor — B0, B1, B2, ours, oracle

| | |
|---|---|
| **Status** | Run on the real model (CPU, Lima VM): `smoke`, `probe` (12 full), `probe_partial`, `probe_partial_gap`, `probe_partial_wide` (8 partial each), the `probe_*` variants, and `main` (50 full, random goal, seeds 1000+; pre-registered in `PREREGISTRATION_main.md`, results in `results/main_primary.md`) |
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
| ours | same trigger as B2 | best-of-N; if all N hit: A* around the blocked cell, junctions and turns become waypoints, one short plan per straight leg. Optional `gap_route` for partial obstacles with no detour (see results) |
| oracle | at t = 0 and at `t_appear` | A* on the known map + speed-limited PD controller (gains untuned) |

The obstacle is a virtual region: the first step on which the ball overlaps it ends the episode as a failure. An episode stops at the first step within `success_radius` of the goal.

Settings shared by all methods: `n_samples`, `check_margin` (extra clearance in the predicted-collision check), and `replan_horizon`. With `"remaining"`, a plan to the goal is shortened so that it ends before the step limit; with `"full"` it is always 384 steps; with `"shrink"` it ends when the t = 0 plan would have ended.

Waypoint legs (ours): with `waypoint_mode: "corners"` every turn of the A* route is a waypoint, so each leg is straight (`r3_horizon` F3: bent legs cut through walls). A waypoint counts as reached when the ball is within `waypoint_radius` of the cell centre (0.5 = inside the cell); a leg is replanned if its plan runs out `leg_grace_steps` before that. Each leg is planned with `n_samples_leg` samples and a horizon of `steps_per_cell` per cell; with `leg_start_at_rest` the start velocity in the inpainted first state is set to zero (the ball still moves; it only changes the conditioning). Among the leg samples that are clear of the obstacle, the loop prefers one that stays on the A* route cells, then one with fewer points inside walls, then one that ends at rest.

## Outputs (`results/<config>/`)

`<method>/episodes/seedNNN.{json,npz,png}` per episode (the JSON has an `events` log: triggers, replans, A* calls, legs). `analyze_blocked.py` writes `episodes.csv`, `summary.json`, `summary.md`, `comparison.png`, and for B2/ours how many of the N samples were clear at the first trigger (risk R2).

## Figures (`make_figures.py`)

`make_figures.py` draws the report figures from saved results only (no model): `bash experiments/common/setup/run.sh python experiments/blocked/make_figures.py` (`--only <name>...`, `--no_anim`). It writes `figures/<name>.png` and `.pdf`; animations are written as GIFs, with a last-frame `<name>_final.png` and `.pdf`. The MP4s are converted from the GIFs on the host: `ffmpeg -i x.gif -fps_mode cfr -r 20 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 20 x.mp4`. The GIFs (7-11 MB each) are not committed.

| Figure | What it shows |
|---|---|
| `fig_pipeline` | the planning loop, coloured by layer (route = A*, motion plan = Diffuser, execution = PD + MuJoCo) |
| `fig_scenario` | main seed 1042: the t = 0 plan runs into the new obstacle; the A* route with junction and corner waypoints |
| `fig_primary` | main, 29 longer detours: success and Wilson CI for all 7 methods, and the paired B2 vs ours (junctions) table (exact McNemar) |
| `fig_by_extra` | the same split by extra detour cells (<= 6 vs >= 8) |
| `fig_outcomes` | every paired scenario x method, sorted by extra cells, coloured by outcome |
| `fig_failures` | failure reasons on the longer detours |
| `fig_cost` | Diffuser planning time per episode (CPU VM) |
| `fig_samples` | `diag/trigger_sampling`: 64 samples at the trigger state for a short and a long detour, and usable samples vs extra cells for H = 384/512/640 |
| `anim_methods` | seed 1042 (12 cells longer): B2, ours (junctions), astar_line + lag, oracle |
| `anim_junction` | seed 1026, (3,3) blocked: junction waypoints hit the obstacle, corner waypoints get through (post-hoc finding 1) |
| `anim_lag` | seed 1046: B2 falls behind a clear plan and stalls on a wall; B1's periodic replanning recovers (post-hoc finding 2) |

In the animations the blue dot is the plan point the controller tracks at that step (`waypoint_action`), so a ball far behind it is lagging its plan.

## Configs

| Config | Purpose |
|---|---|
| `smoke.json` | 2 full + 1 partial scenario, N = 4: does the pipeline run |
| `probe.json` | 12 full scenarios, N = 16: R2 probe and the single-scenario detour check |
| `probe_partial.json` | 8 partial scenarios (gap 0.5), N = 16 |
| `probe_partial_gap.json` | the same 8 scenarios, ours with `gap_route` |
| `probe_partial_wide.json` | 8 partial scenarios with gap 0.7 (`partial_frac` 0.3) |
| `main.json` | 50 full blockages, random goal per seed, seeds from 1000 (pre-registered); `main_junctions.json` and `main_lag_line.json` reuse its scenarios |

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

Seed 4 is dropped from every method: the oracle moves faster than the t = 0 plan and was already inside the cell when the obstacle appeared, so its episode is `excluded` (`obstacle_appeared_on_agent`). 7 paired scenarios remain. 4 of the 5 scenarios without a detour block the same cell, (6,8), the last corridor before the goal.

| Method | All 7 | Detour exists (3) | No detour: must pass the gap (4) |
|---|---|---|---|
| B0 | 1/7 | 1/3 | 0/4 |
| B1 | 1/7 | 1/3 | 0/4 |
| B2 | 1/7 | 1/3 | 0/4 |
| **ours** | 3/7 | **3/3** | 0/4 |
| oracle | 7/7 | 3/3 | 4/4 |

- **The proposal's step 2 ("best-of-N finds a plan through the gap") does not hold at gap = 0.5**: at the first trigger, a median of 0 of 16 samples was clear (mean 0.5 when no detour exists). The model's plans run along the middle of the corridor, which the obstacle covers.
- When a detour exists: on seeds 6 and 7 all 16 samples hit, so ours treated the cell as blocked and took the 16- and 21-cell detours; on seed 1 a same-length route existed and best-of-N found it (12 of 16 clear). Ours: 3/3.
- When no detour exists, nobody but the oracle passes the gap. Ours falls back to the B2 behaviour (same outcome on every seed): on seeds 0 and 2 it adopted the only clear sample and the ball's tracking error carried it into the obstacle; on seeds 3 and 4 the adopted plans toured and stalled on a wall; on seed 5 no sample was ever clear and the ball ran into the obstacle.

### Passing the gap on waypoint legs (`probe_partial_gap`, ours only)

`gap_route: true` (off by default): when the obstacle is partial and no route avoids its cell, ours skips best-of-N to the goal and routes through the cell. The cells before and after it become waypoints moved sideways onto the centre line of the gap, as in the oracle's route, and the ball has to come within `gap_waypoint_radius` (0.1) of them. Same 8 scenarios (`scenarios.json` and `plans0.npz` copied from `probe_partial`).

- No detour: **0/5** (4 timeouts, 1 obstacle hit; without `gap_route`: 3 hits, 2 stalls). Seeds 1, 6 and 7, which do not use the gap route, reproduce `probe_partial` step for step.
- The ball never settles within 0.1 of the gap entry point (seeds 2, 3, 5): each 32-step leg plan ends, the leg is planned again 20 steps later, and this repeats until the step limit. Near (5,8) the leg plans curve east toward column 9, the same behaviour as the leg (5,8)->(7,8) that failed in `probe` (seed 10). On seed 0 the ball reached the entry point, but 0 of 8 samples of the leg through the gap were clear.
- So the positioning the gap needs (about +-0.12 around the centre line for a 0.5 gap with `check_margin` 0.05) is finer than what the frozen model and the tracking controller deliver. A 0.5 gap with no detour is out of reach for this planning loop as it stands.

### Wider gap (`probe_partial_wide`, gap = 0.7, `gap_route` off)

Same seeds and cells as `probe_partial`; only `partial_frac` changes (0.3 instead of 0.5). No seed is excluded.

| Method | All 8 | Detour exists (3) | No detour (5) |
|---|---|---|---|
| B0 | 2/8 | 1/3 | 1/5 |
| B2 | 2/8 | 1/3 | 1/5 |
| **ours** | 4/8 | **3/3** | 1/5 |
| oracle | 8/8 | 3/3 | 5/5 |

- With the wider gap, best-of-N finds a clear plan: at the first trigger on the no-detour seeds, 4 of 5 had at least one clear sample (mean 4.4 of 16; at gap 0.5: 2 of 5, mean 0.4). So the proposal's step 2 works **at the plan level** once the gap is 0.7.
- It fails in **execution**: 3 of those 4 clear plans still end in an obstacle hit (seeds 0, 2, 3). The adopted plan clears the obstacle by 0.20-0.37, but the ball runs 50-120 steps behind it (the plan moves about 0.044 per step and the ball about 0.03). The controller (the original `plan_maze2d.py` one) tracks the plan point of the current time step, so a lagging ball heads straight for a point far ahead and cuts the inside of the turn just before the obstacle cell: 0.4-0.5 off the plan's path at its closest approach. On seed 4, the only success, the lag was 3 steps.
- This is a different failure from `probe` seed 11 (full blockage), where the ball was 0.19 off the path with no lag.
- Hence the predicted-collision check tests the plan, not the path the ball will take. Candidates (not tried): also trigger a replan when the ball falls more than a few steps behind its plan, or check the plan with a margin that grows with that lag.

### Not working (tried, kept out of the method)

- Interior inpainting (proposal option b, fix a via point mid-plan instead of chaining legs): 0 of 48 samples rerouted; the plans jump about 4.3 units to the via point and back (`.vm/diag/interior_probe`).
- Conditioning a leg on the ball's actual velocity: 18 of 48 leg samples clean, against 24 of 48 when the start is inpainted at rest (`.vm/diag/leg_velocity_probe`). Hence `leg_start_at_rest`.
