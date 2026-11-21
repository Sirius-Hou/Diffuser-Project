# main: pre-registered analysis (written 2026-11-21, before any main run)

Agreed by Ken on 2026-11-21. Anything outside this list is reported as post-hoc.

## Scenarios (`configs/main.json`)
- 50 full blockages, no partial ones (partial blockages are reported from the probes as a limitation).
- Random goal per seed, at least 8 cells from the start (`random_goal`, `min_goal_cells: 8`).
- `seed_start: 1000`: no overlap with any probe (seeds 0-15).
- Short detours and same-length scenarios are kept and reported separately.
- Same settings as the probes except `max_triggers` 10 (probe: 5; identical results on `probe_multigoal`).
- `main` (B0, B1, B2, ours corners, oracle), `main_junctions` (ours, junction waypoints) and `main_lag_line` (astar_line, `lag_trigger` 1.0) share one `scenarios.json` / `plans0.npz`.

## Primary comparison (the only one)
- Population: full blockage, detour longer than the route (`case == "full, longer detour"`), not excluded by the oracle.
- B2 vs ours (junctions), paired success at radius 0.5.
- Two-sided exact McNemar test on the discordant pairs, alpha = 0.05; also each success rate with a Wilson 95% CI.
- Prediction (from `probe_multigoal`, 9/12 vs 5/12): ours (junctions) > B2.

## Secondary (descriptive only)
1. Success of B0, B1, ours (corners), astar_line + lag trigger, oracle.
2. B2 success split by extra detour cells: <= 6 and >= 8. Prediction: close to 0 for >= 8.
3. Success on same-length scenarios.
4. Success at radius 0.75 and 1.0 (from `min_dist_to_goal`).
5. How many scenarios use A*, and which cells are blocked.
