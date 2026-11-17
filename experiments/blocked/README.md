# Blocked corridor — B0, B1, B2, ours, oracle

| | |
|---|---|
| **Status** | Code written; checked only with a fake planner (no MuJoCo/torch on the dev machine). Not run on the real model yet |
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
| ours | same trigger as B2 | best-of-N; if all N hit: A* around the blocked cell, junction cells become waypoints, one short plan per leg |
| oracle | at t = 0 and at `t_appear` | A* on the known map + speed-limited PD controller (gains untuned) |

The obstacle is a virtual region: the first step on which the ball overlaps it ends the episode as a failure. An episode stops at the first step within `success_radius` of the goal.

Settings shared by all methods: `n_samples`, `check_margin` (extra clearance in the predicted-collision check), and `replan_horizon`. With `"remaining"`, a plan to the goal is shortened so that it ends before the step limit; with `"full"` it is always 384 steps.

## Outputs (`results/<config>/`)

`<method>/episodes/seedNNN.{json,npz,png}` per episode (the JSON has an `events` log: triggers, replans, A* calls, legs). `analyze_blocked.py` writes `episodes.csv`, `summary.json`, `summary.md`, `comparison.png`, and for B2/ours how many of the N samples were clear at the first trigger (risk R2).

## Configs

| Config | Purpose |
|---|---|
| `smoke.json` | 2 full + 1 partial scenario, N = 4: does the pipeline run |
| `probe.json` | 12 full scenarios, N = 16: R2 probe and the single-scenario detour check |
| `main.json` | 35 full + 15 partial scenarios |

`steps_per_cell`, `min_horizon` and `max_leg_cells` are placeholders until `r3_horizon` has run.
