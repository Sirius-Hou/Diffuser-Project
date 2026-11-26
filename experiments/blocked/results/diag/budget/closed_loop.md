# Closed-loop evidence for the time-budget question (post hoc, exploratory)

Route left = BFS cells from the ball's cell at t_appear (B0 rollout) to the goal with the obstacle cell blocked.

## main: longer detour (n = 28; dropped [1005, 1007, 1013, 1039])

| method | 0-12 | 13-15 | 16-18 | >= 19 | <= 15 vs >= 16 | Fisher p |
|---|---|---|---|---|---|---|
| B0 | 0/7 | 0/10 | 0/7 | 0/4 | 0/17 vs 0/11 | 1.000 |
| B1 | 3/7 | 7/10 | 3/7 | 1/4 | 10/17 vs 4/11 | 0.440 |
| B2 | 4/7 | 4/10 | 1/7 | 0/4 | 8/17 vs 1/11 | 0.049 |
| ours (corners) | 7/7 | 9/10 | 6/7 | 2/4 | 16/17 vs 8/11 | 0.269 |
| ours (junctions) | 5/7 | 9/10 | 5/7 | 3/4 | 14/17 vs 8/11 | 0.653 |
| astar_line + lag | 7/7 | 10/10 | 7/7 | 4/4 | 17/17 vs 11/11 | 1.000 |
| oracle | 7/7 | 10/10 | 7/7 | 4/4 | 17/17 vs 11/11 | 1.000 |

## main: all (n = 46; dropped [1005, 1007, 1013, 1039])

| method | 0-12 | 13-15 | 16-18 | >= 19 | <= 15 vs >= 16 | Fisher p |
|---|---|---|---|---|---|---|
| B0 | 7/23 | 0/11 | 0/8 | 0/4 | 7/34 vs 0/12 | 0.165 |
| B1 | 14/23 | 7/11 | 3/8 | 1/4 | 21/34 vs 4/12 | 0.107 |
| B2 | 19/23 | 5/11 | 1/8 | 0/4 | 24/34 vs 1/12 | 0.000 |
| ours (corners) | 23/23 | 10/11 | 6/8 | 2/4 | 33/34 vs 8/12 | 0.013 |
| ours (junctions) | 21/23 | 10/11 | 5/8 | 3/4 | 31/34 vs 8/12 | 0.064 |
| astar_line + lag | 23/23 | 11/11 | 8/8 | 4/4 | 34/34 vs 12/12 | 1.000 |
| oracle | 23/23 | 11/11 | 8/8 | 4/4 | 34/34 vs 12/12 | 1.000 |

## full_fresh: longer detour (n = 23; dropped none)

| method | 0-12 | 13-15 | 16-18 | >= 19 | <= 15 vs >= 16 | Fisher p |
|---|---|---|---|---|---|---|
| B0 | 0/8 | 0/9 | 0/4 | 0/2 | 0/17 vs 0/6 | 1.000 |
| B1 | 4/8 | 3/9 | 0/4 | 0/2 | 7/17 vs 0/6 | 0.124 |
| B2 | 4/8 | 2/9 | 1/4 | 0/2 | 6/17 vs 1/6 | 0.621 |
| B2 + lag replan | 5/8 | 5/9 | 1/4 | 0/2 | 10/17 vs 1/6 | 0.155 |
| oracle | 8/8 | 9/9 | 4/4 | 2/2 | 17/17 vs 6/6 | 1.000 |

## full_fresh: all (n = 30; dropped none)

| method | 0-12 | 13-15 | 16-18 | >= 19 | <= 15 vs >= 16 | Fisher p |
|---|---|---|---|---|---|---|
| B0 | 2/15 | 0/9 | 0/4 | 0/2 | 2/24 vs 0/6 | 1.000 |
| B1 | 8/15 | 3/9 | 0/4 | 0/2 | 11/24 vs 0/6 | 0.061 |
| B2 | 9/15 | 2/9 | 1/4 | 0/2 | 11/24 vs 1/6 | 0.358 |
| B2 + lag replan | 10/15 | 5/9 | 1/4 | 0/2 | 15/24 vs 1/6 | 0.072 |
| oracle | 15/15 | 9/9 | 4/4 | 2/2 | 24/24 vs 6/6 | 1.000 |

## Can the closed loop separate total length from extra cells? (main + full_fresh, longer detours, B2)

| | extra <= 6 | extra >= 8 |
|---|---|---|
| total <= 15 | 13/31 | 1/3 |
| total >= 16 | 0/3 | 2/14 |

## ours: legs

### ours (corners): 272 leg plans

- leg length (cells): max 4, horizon max 128
- steps per cell to reach the waypoint: median 27.0 (p10 21.5, p90 65.0; n = 173)

| leg | plans | scenarios | samples clear |
|---|---|---|---|
| shortest route to the waypoint open | 272 | 26 | 2134/2176 = 98.1% |

### ours (junctions): 205 leg plans

- leg length (cells): max 10, horizon max 320
- steps per cell to reach the waypoint: median 29.5 (p10 24.0, p90 62.6; n = 87)

| leg | plans | scenarios | samples clear |
|---|---|---|---|
| shortest route to the waypoint open | 144 | 26 | 1111/1152 = 96.4% |
| shortest route to the waypoint blocked | 61 | 6 | 8/488 = 1.6% |

