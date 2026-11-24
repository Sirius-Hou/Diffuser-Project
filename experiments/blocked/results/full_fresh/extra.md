# full_fresh: follow-up experiment (exploratory, 2026-11-23)

Paired scenarios: 30 (dropped: none)

| Method | longer detour (n=23) | same length (n=7) | all (n=30) | hit obstacle | planning time (s, mean) |
|---|---|---|---|---|---|
| B0 | 0/23 = 0% (-0-14%) | 2/7 = 29% (8-64%) | 2/30 = 7% (2-21%) | 27/30 | 5 |
| B1 | 7/23 = 30% (16-51%) | 4/7 = 57% (25-84%) | 11/30 = 37% (22-54%) | 14/30 | 132 |
| B2 | 7/23 = 30% (16-51%) | 5/7 = 71% (36-92%) | 12/30 = 40% (25-58%) | 15/30 | 208 |
| B2_lag | 11/23 = 48% (29-67%) | 5/7 = 71% (36-92%) | 16/30 = 53% (36-70%) | 12/30 | 238 |
| oracle | 23/23 = 100% (86-100%) | 7/7 = 100% (65-100%) | 30/30 = 100% (89-100%) | 0/30 | 0 |

## Paired comparisons (exact McNemar, two-sided)

- B2 vs B2_lag (longer detour, n=23): only B2 0, only B2_lag 4; p = 0.1250
- B1 vs B2_lag (longer detour, n=23): only B1 2, only B2_lag 6; p = 0.2891
- B1 vs B2 (longer detour, n=23): only B1 4, only B2 4; p = 1.0000
- B2 vs B2_lag (all, n=30): only B2 0, only B2_lag 4; p = 0.1250

## Failure reasons and where obstacle collisions came from

- B0: failures {'obstacle_collision': 27, 'stuck_on_wall': 1}; collisions by layer {'no replan': 27}
- B1: failures {'obstacle_collision': 14, 'timeout_still_moving': 5}; collisions by layer {'planning': 13, 'execution': 1}
- B2: failures {'stuck_on_wall': 3, 'obstacle_collision': 15}; collisions by layer {'execution': 6, 'planning': 9}
- B2_lag: failures {'timeout_still_moving': 2, 'obstacle_collision': 12}; collisions by layer {'execution': 3, 'planning': 9}; lag replans 28
- oracle: failures none; collisions by layer none

## Per scenario

| seed | blocked | longer detour | same length | B0 | B1 | B2 | B2_lag | oracle |
|---|---|---|---|---|---|---|---|---|
| 4000 | 5,8 |  | x | obstacle_collision | obstacle_collision | ok | ok | ok |
| 4001 | 3,6 | x |  | obstacle_collision | ok | ok | ok | ok |
| 4002 | 5,6 | x |  | obstacle_collision | ok | stuck_on_wall | timeout_still_moving | ok |
| 4003 | 1,6 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 4004 | 3,3 | x |  | obstacle_collision | timeout_still_moving | obstacle_collision | obstacle_collision | ok |
| 4005 | 5,7 | x |  | obstacle_collision | timeout_still_moving | obstacle_collision | obstacle_collision | ok |
| 4006 | 3,3 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 4007 | 3,3 | x |  | obstacle_collision | ok | ok | ok | ok |
| 4008 | 1,6 | x |  | obstacle_collision | timeout_still_moving | ok | ok | ok |
| 4009 | 6,2 |  | x | ok | ok | ok | ok | ok |
| 4010 | 3,2 | x |  | obstacle_collision | ok | obstacle_collision | obstacle_collision | ok |
| 4011 | 5,7 | x |  | obstacle_collision | obstacle_collision | stuck_on_wall | ok | ok |
| 4012 | 5,7 | x |  | obstacle_collision | obstacle_collision | ok | ok | ok |
| 4013 | 3,3 |  | x | obstacle_collision | ok | ok | ok | ok |
| 4014 | 3,3 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 4015 | 5,7 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 4017 | 3,2 | x |  | obstacle_collision | ok | obstacle_collision | ok | ok |
| 4018 | 7,1 |  | x | ok | ok | ok | ok | ok |
| 4019 | 4,6 | x |  | obstacle_collision | obstacle_collision | ok | ok | ok |
| 4020 | 3,3 | x |  | stuck_on_wall | obstacle_collision | obstacle_collision | ok | ok |
| 4022 | 2,6 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 4023 | 2,6 | x |  | obstacle_collision | obstacle_collision | ok | ok | ok |
| 4024 | 3,2 | x |  | obstacle_collision | ok | obstacle_collision | ok | ok |
| 4025 | 2,1 |  | x | obstacle_collision | timeout_still_moving | ok | ok | ok |
| 4026 | 1,1 |  | x | obstacle_collision | ok | obstacle_collision | obstacle_collision | ok |
| 4027 | 5,7 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 4028 | 1,8 | x |  | obstacle_collision | ok | ok | ok | ok |
| 4029 | 4,6 |  | x | obstacle_collision | obstacle_collision | stuck_on_wall | timeout_still_moving | ok |
| 4030 | 3,1 | x |  | obstacle_collision | timeout_still_moving | obstacle_collision | obstacle_collision | ok |
| 4031 | 4,6 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | ok |
