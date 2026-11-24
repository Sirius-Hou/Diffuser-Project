# partial_exec: follow-up experiment (exploratory, 2026-11-23)

Paired scenarios: 16 (dropped: none)

| Method | another route (n=11) | no other route (n=5) | all (n=16) | hit obstacle | planning time (s, mean) |
|---|---|---|---|---|---|
| oracle | 10/11 = 91% (62-98%) | 3/5 = 60% (23-88%) | 13/16 = 81% (57-93%) | 3/16 | 0 |
| B2 | 4/11 = 36% (15-65%) | 1/5 = 20% (4-62%) | 5/16 = 31% (14-56%) | 9/16 | 163 |
| b2_guided | 4/11 = 36% (15-65%) | 0/5 = 0% (-0-43%) | 4/16 = 25% (10-49%) | 12/16 | 189 |
| B2 + path | 4/11 = 36% (15-65%) | 1/5 = 20% (4-62%) | 5/16 = 31% (14-56%) | 8/16 | 147 |
| b2_guided + path | 3/11 = 27% (10-57%) | 1/5 = 20% (4-62%) | 4/16 = 25% (10-49%) | 8/16 | 230 |

## Paired comparisons (exact McNemar, two-sided)

- b2_guided vs b2_guided + path (all, n=16): only b2_guided 1, only b2_guided + path 1; p = 1.0000
- B2 vs B2 + path (all, n=16): only B2 1, only B2 + path 1; p = 1.0000
- B2 + path vs b2_guided + path (all, n=16): only B2 + path 1, only b2_guided + path 0; p = 1.0000
- B2 vs b2_guided (all, n=16): only B2 2, only b2_guided 1; p = 1.0000

## Failure reasons and where obstacle collisions came from

- oracle: failures {'obstacle_collision': 3}; collisions by layer {'no replan': 3}
- B2: failures {'obstacle_collision': 9, 'stuck_on_wall': 2}; collisions by layer {'planning': 8, 'execution': 1}
- b2_guided: failures {'obstacle_collision': 12}; collisions by layer {'execution': 9, 'planning': 3}
- B2 + path: failures {'stuck_on_wall': 1, 'obstacle_collision': 8, 'timeout_still_moving': 2}; collisions by layer {'planning': 7, 'execution': 1}
- b2_guided + path: failures {'obstacle_collision': 8, 'timeout_still_moving': 4}; collisions by layer {'execution': 5, 'planning': 3}

## Per scenario

| seed | blocked | another route | no other route | oracle | B2 | b2_guided | B2 + path | b2_guided + path |
|---|---|---|---|---|---|---|---|---|
| 5000 | 3,5 |  | x | obstacle_collision | obstacle_collision | obstacle_collision | stuck_on_wall | obstacle_collision |
| 5001 | 4,6 | x |  | ok | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 5002 | 4,10 | x |  | ok | stuck_on_wall | ok | ok | timeout_still_moving |
| 5003 | 2,6 | x |  | ok | obstacle_collision | obstacle_collision | obstacle_collision | timeout_still_moving |
| 5004 | 4,1 |  | x | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 5005 | 3,5 |  | x | ok | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 5006 | 4,6 | x |  | ok | ok | ok | ok | ok |
| 5007 | 4,6 | x |  | ok | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 5008 | 2,6 | x |  | ok | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 5009 | 3,3 | x |  | ok | ok | obstacle_collision | obstacle_collision | obstacle_collision |
| 5010 | 2,8 | x |  | ok | ok | ok | ok | ok |
| 5011 | 3,3 | x |  | obstacle_collision | obstacle_collision | obstacle_collision | timeout_still_moving | timeout_still_moving |
| 5012 | 6,6 |  | x | ok | ok | obstacle_collision | ok | ok |
| 5013 | 4,6 | x |  | ok | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 5014 | 7,5 | x |  | ok | ok | ok | ok | ok |
| 5015 | 7,9 |  | x | ok | stuck_on_wall | obstacle_collision | timeout_still_moving | timeout_still_moving |
