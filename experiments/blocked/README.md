# Blocked corridor — B0 / B1 / B2 / Ours on a frozen scenario list

| | |
|---|---|
| **Status** | Done. All runs on Colab T4 GPU (configs use `device: cuda`). Results in `results/` (json/csv/overview only; per-episode plots kept on Google Drive). |
| **Project steps** | 2 (obstacle, B0 blocked, B2 probe) and 3 (predictive trigger + best-of-N + A* chaining, B1, scenario list) |
| **Needs** | the same runtime as `b0_baseline` (Linux/WSL/Colab). Shared code tests run anywhere |

## 1. What is new

### Shared code (`experiments/common/`, numpy only unless noted)

| File | Team interface / purpose |
|---|---|
| `mazes.py` | Large maze as a numpy array (copied from d4rl), so geometry code runs without mujoco. Runners assert it equals `env.maze_arr` |
| `obstacles.py` | `Obstacle` (full cell, or partial slab across a corridor), `first_collision(traj, obstacles)` = **會不會撞(軌跡, 障礙物)**, `ball_collides` (executed ball, every step), `blocked_cells`, drawing |
| `waypoints.py` | `astar(start_cell, goal_cell, blocked)` → waypoint list = **astar(起點格, 終點格, 被擋的格)**; `segment_horizon` |
| `planner.py` | `plan(obs, goal, n, horizon=None)`: **new `horizon` argument**. Default = 384 = exactly the old call, so B0 is unchanged |
| `rollout.py` | the one episode loop every method uses: seeding/reset (B0 order), obstacle appearance, ball-vs-obstacle check, controller, success, metrics, per-episode plot |
| `methods.py` | `B0`, `B1`, `B2`, `Ours`, `Oracle`: they only decide when to replan and which plan to follow |
| `fakes.py` | fake env + fake planner for logic tests on machines without mujoco (numbers mean nothing) |
| `test_common.py` | `python experiments/common/test_common.py` → `ALL 8 TESTS PASSED` |

### This experiment (`experiments/blocked/`)

| File | Purpose |
|---|---|
| `make_scenarios.py` + `configs/scenarios.json` | builds the frozen scenario list (run once, commit the output) |
| `probe_b2.py` + `configs/probe_b2.json` | risk R2: how many of 16 plans already avoid the blockage at B2's first check; also plan time for n = 1/4/16 (R5) |
| `run_blocked.py` + `configs/{smoke,main}.json` | runs methods on scenarios; resumes after a disconnect (skips finished episodes) |
| `analyze_blocked.py` | `episodes.csv`, `summary.json` (success + Wilson CI, collisions, cost, paired Ours-vs-B2), `overview.png` |

## 2. Design decisions (change them in configs, not in code)

**Obstacle.** Purely logical, not added to MuJoCo. The ball is checked every step (disc radius 0.1 vs box); touching it = collision = episode ends as a failure. Appears at `appear_t`, then fixed. Known from that step (perfect sensing).

**All methods start identical.** Same seed → same start → same first plan (n = 1, H = 384), so every method follows the B0 plan until its first replan.

**Plan check.** Remaining plan = ball position + the rest of the current plan (densified to 0.05). It "collides" if any point is within `0.1 + collision_margin` (default 0.05) of a known obstacle. Walls are not checked here (proposal).

**best-of-N selection.** A plan is OK if it avoids every known obstacle **and does not tunnel through a wall** (no point deeper than `max_wall_depth` = 0.2 inside a wall; corner-cutting, typically < 0.1, is allowed). Take the shortest OK plan. If none is OK (counted as `n_all_collide`; this is what sends Ours to A*), follow the least-bad one: clear of the obstacle with the shallowest wall crossing, else the one whose first collision is latest.

**Margin near the obstacle.** If the ball is already inside the 0.05 safety margin (but not touching), plan checks use the bare ball radius, otherwise every plan starting at the ball would count as colliding at its first point.

*Why (smoke run, `full_s003`, real model):* the first version accepted any obstacle-free plan. With a short horizon the only "clear" plans went straight through the wall below (5,10); the ball pushed against that wall for ~190 steps (the trigger does not look at walls), then all plans were rejected because the ball sat inside the margin, and it hit the obstacle.

| Method | When | What |
|---|---|---|
| B0 | never | follow the t = 0 plan |
| B1 | every `k_periodic` = 100 steps | best-of-N to the goal |
| B2 | every `k` = 10 steps, only if the remaining plan hits a known obstacle | best-of-N to the goal; if none OK, least-bad |
| Oracle (reference) | when the obstacle appears | drop the Diffuser: A* on the known map + D4RL's PD waypoint controller (p = 10, d = −1). Before the obstacle it follows the same first plan, so it reacts from the same state as the others |
| Ours | same trigger as B2 | best-of-N; if none of the N is OK → A* without the obstacle cell → waypoints → one plan per segment (best-of-N each). Next segment when within 0.5 of the waypoint or the segment plan is used up (+20 steps) |

**Waypoints (`waypoint_mode`).** `greedy` (default): the fewest waypoints such that, for every segment, *no* shortest path of the unblocked maze passes the obstacle (Diffuser imitates near-shortest routes of the unblocked maze). `junctions`: every junction on the A* route (the proposal's literal wording), for comparison.

**Horizon — important (B0 README F1).** Diffuser stretches every plan over its whole horizon: with H = 384 it arrives ~360 steps later, whatever the distance. So:
- a replan with H = 384 after t ≈ 440 cannot finish in 800 steps;
- periodic replanning (B1) with H = 384 keeps pushing the arrival back and may never arrive.

Therefore replans to the goal use `replan_horizon: "deadline"` by default for **all** methods (fair, map-free): `H = clip(round4(800 − t − 40), 64, 384)`. Early replans get the full 384 steps (room for a detour); late replans shrink so the plan still ends before the step limit. Segments of Ours use `segment_horizon(A* cells) = clip(round4(30 · cells + 32), 64, 384)`. H ≠ 384 was never seen in training.

*Why not `"euclid"`* (horizon from the straight-line distance, the first default): the B2 probe showed that with it almost no plan switches corridor; the detour does not fit in the time budget, so plans either run into the obstacle or tunnel through a wall. That would handicap B1/B2 for a reason unrelated to the method. `"euclid"`, `"full"` (always 384) and integers remain available.

**Waypoint guidance (`guidance`, Ours only).** `chain` (default, main results): one Diffuser plan per segment. `inpaint` (ablation, proposal contribution 3): ONE plan from the ball to the goal; the waypoint's x-y is pinned at a middle step k (velocity left free), k proportional to the waypoint's distance along the A* route; horizon from the A* length. Implemented in `planner.plan(..., via=[(k, (x, y))])` by swapping `apply_conditioning` only during that call (weights and model code unchanged). Each `inpaint` event logs n_ok, n_clear, n_wall_ok and n_kink (plans with a step > 4x the median step near k). Config: `configs/ablation_inpaint.json`.

**Waypoint velocity.** Segment end is conditioned like the goal: `[x, y, 0, 0]`, so the ball slows at each waypoint (costs steps; only-x-y inpainting is the optional variant).

**Scenario filter** (`configs/scenarios.json`): obstacle on a corridor cell (two opposite free neighbours) of the ball's actual B0 route; appears at t ∈ [50, 250]; the ball reaches the cell 30–200 steps later; a detour exists and is ≥ 2 cells longer than the unblocked path; `appear_t + 10 + 30 · detour cells + 80 ≤ 800`. Seeds from 0, so seeds 0–29 can be compared with the obstacle-free B0 run. Each full scenario also gets a `partial50` copy (half the corridor blocked) for later; configs run `kinds: ["full"]` only.

## 3. Order of runs

```bash
python experiments/common/test_common.py                                             # anywhere
python experiments/blocked/make_scenarios.py --config experiments/blocked/configs/scenarios.json
python experiments/blocked/probe_b2.py       --config experiments/blocked/configs/probe_b2.json
python experiments/blocked/run_blocked.py    --config experiments/blocked/configs/smoke.json
python experiments/blocked/analyze_blocked.py experiments/blocked/results/smoke
python experiments/blocked/run_blocked.py    --config experiments/blocked/configs/main.json
python experiments/blocked/analyze_blocked.py experiments/blocked/results/main
```
(Inside WSL prefix each with `bash experiments/common/setup/run.sh`.) Add `--fake` to any of them to test the logic without mujoco.

**Same device for everything.** The scenario file depends on the exact B0 rollout. Generate scenarios and run all methods on the same device type (all CPU, or all the same GPU); otherwise the obstacle may no longer lie on the ball's path. Colab CPU takes ~11 s per plan (vs 2 s on an 8-thread desktop), so a full main run on Colab CPU takes hours; Colab T4 GPU is much faster (check repro there first).

## 4. Results (Colab T4, real model, 2026-11-11)

**Scenario list:** 50 full-blockage scenarios (+ 50 `partial50` copies), from seeds 0–113 (skipped: 50 no valid cell, 14 B0 fails without obstacle). Detours +4 to +12 cells, appear t = 50–205, one A* waypoint each. Obstacles concentrate on a few cells ((4,6), (5,7), (5,9)) because the goal is fixed: most other corridors are bridges with no detour. Variety comes from start, appear time and ball position — state this as a limitation.

**B2 probe (R2), first 20 scenarios, N = 16, at B2's first check** (valid = avoids the obstacle and no wall tunnelling):

| horizon | scenarios with ≥ 1 valid plan among 16 |
|---|---|
| full (384) | 4 / 20 (20 %) |
| deadline (used in the main run; = 384 here) | 6 / 20 (30 %) |
| euclid | 2 / 20 (10 %) |

→ R2 did not materialise. full vs deadline differ only by sampling noise. Plan time on T4: n = 1, 4, 16 all ≈ 3.7 s.

**Main run, full blockage, 50 scenarios** (`configs/main.json`):

| method | success | 95 % CI | obstacle collision | plans / episode | plan s / episode | failures |
|---|---|---|---|---|---|---|
| B0 | 0 % | 0–7 % | 100 % | 1 | 3.7 | collision 50 |
| B1 | 34 % | 22–48 % | 34 % | 100.5 | 26.1 | collision 17, timeout 16 |
| B2 | 20 % | 11–33 % | 12 % | 194.3 | 46.2 | stuck on wall 22, timeout 12, collision 6 |
| **Ours** | **76 %** | **63–86 %** | **0 %** | **53.8** | **15.4** | timeout 9, stuck on wall 3 |
| Oracle | 100 % | 93–100 % | 0 % | 1 | 3.6 | — |

Ours vs B2, paired: both 10, **only Ours 28, only B2 0**, neither 12 (McNemar exact p ≈ 7e-9).

**Partial blockage, 50 scenarios** (`configs/partial.json`, half the corridor blocked, gap 0.5): B0 46 %, B1 40 %, B2 36 %, **Ours 70 %** (collision 2 %), Oracle 100 %; Ours vs B2 paired: only Ours 18, only B2 1. Re-planning with the Diffuser alone (B1, B2) is worse than not re-planning: the trigger is conservative (false alarms) and new plans detour, time out or get stuck. Only the map-based reroute improves on B0.

**Ablation 1 — which waypoints** (`configs/ablation_junctions.json`, Ours only, full blockage): a waypoint at every junction gives 62 % (collision 30 %, 107 plans/episode) vs greedy 76 % (collision 0 %, 54). All 15 collisions have a waypoint on the cell next to the obstacle (almost always (5,8)): the ball switches segment and turns there, overshoots and grazes the obstacle.

**Ablation 2 — how waypoints are used** (`configs/ablation_inpaint.json`, Ours only, full blockage; proposal contribution 3): middle inpainting gives 20 % (212 plans/episode). Of 9,728 sampled plans, 98.8 % show an isolated spike at the pinned step (jump to the waypoint and back), only 1.4 % are usable. The frozen Diffuser does not absorb middle conditions it never saw in training; subgoal chaining only pins segment ends, as in training.

**Figures:** `figures/fig1_success.png` (success, both blockage types), `fig2_cost.png` (success vs samples), `fig3_failures.png` (outcome breakdown), `b2_full_s004.png` / `ours_full_s004.png` (slide comparison).

**Failures that end 1.8 from the goal** (ball stuck at the wall cell (6,9) just above the goal: Diffuser's own end-of-plan problem, B0 README F2, unrelated to the obstacle): B2 16 of 40, Ours 5 of 12.

**Ours' 12 failures:** all switched corridor and none hit the obstacle. 5 stuck at (6,9) as above (2 of them never needed A*); 3 stuck below the waypoint (3,9) — the second segment's best-of-N found no OK plan and the trigger does not watch walls; 4 ran out of time on long detours (one ended 0.51 from the goal, radius 0.5).

**Figure for slides:** `full_s004` (B2 follows a plan through the wall at (4,9) and gets stuck; Ours goes through the A* waypoint (3,9) and succeeds). Not `full_s011` (B2 fails at (6,9), the Diffuser issue).

## 5. Known limitations
- Short horizons (< 384) and segment endpoints with zero velocity are untested with the real model.
- The fake env/planner only check the wiring; their success rates mean nothing.
- `visited_blocked_cell` and collision are checked on recorded positions every step; a ball faster than ~0.2 per step could skip a corner (not seen in B0, ≤ 0.05 per step).
