# R3 / R5 — short planning horizons and batch cost

| | |
|---|---|
| **Status** | Code written, not run yet |
| **Project step** | Risks R3 (subgoal chaining runs out of time) and R5 (compute) |
| **Run everything** | `bash experiments/r3_horizon/run_all.sh` |

## Why

B0 showed that a 384-step plan arrives at about step 360 whatever the distance (B0 README, F1). Two full-length legs do not fit in an 800-step episode, so each waypoint leg needs a shorter horizon. The model was trained only with H = 384.

What the code allows: the Large U-Net (`dim_mults = (1, 4, 8)`) downsamples twice and has no positional embedding along the plan, so any horizon that is a multiple of 4 runs. `FrozenDiffuserPlanner.plan(..., horizon=h)` calls the unchanged sampler with a shorter sequence. Whether those plans are any good is what this experiment measures.

## What it does

1. **Horizon sweep.** For start/goal cell pairs 1 to 10 cells apart, sample plans with H from 32 to 384, execute each with the original controller, and record whether the ball reaches the goal cell centre (0.3 for a waypoint, 0.5 for the goal), when, and whether the plan cuts through walls.
2. **Batch timing.** Wall-clock time of `plan(n, H)` for n = 1, 4, 8, 16.

## Outputs (`results/<run>/`)

| File | Content |
|---|---|
| `horizon_runs.csv` | one row per sampled plan |
| `horizon_table.csv`, `horizon_reach.png` | reach rate per (cells, horizon) |
| `timing.csv` | seconds per `plan(n, H)` call |
| `summary.json` | `recommendation.steps_per_cell` and `recommendation.min_horizon` |

Copy `steps_per_cell` and `min_horizon` into `experiments/blocked/configs/*.json` (they hold placeholders of 30 and 32 until this has run), then rebuild the scenarios.
