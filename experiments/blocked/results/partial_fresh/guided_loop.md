# partial_fresh: B2 vs b2_guided (exploratory)

Paired scenarios: 15 (dropped: [3008])

| Method | all (n=15) | another route (n=9) | no other route (n=6) | hit obstacle | planning time (s, mean) |
|---|---|---|---|---|---|
| B0 | 0/15 = 0% (-0-20%) | 0/9 = 0% (0-30%) | 0/6 = 0% (0-39%) | 15/15 | 7 |
| B2 | 5/15 = 33% (15-58%) | 5/9 = 56% (27-81%) | 0/6 = 0% (0-39%) | 9/15 | 186 |
| b2_guided | 6/15 = 40% (20-64%) | 4/9 = 44% (19-73%) | 2/6 = 33% (10-70%) | 7/15 | 206 |
| oracle | 8/15 = 53% (30-75%) | 6/9 = 67% (35-88%) | 2/6 = 33% (10-70%) | 7/15 | 0 |

Discordant pairs: only B2 1, only b2_guided 2; exact McNemar two-sided p = 1.0000.

## Where the obstacle collisions came from

- B2: planning 9, execution 0, no replan before the hit 0
- b2_guided: planning 2, execution 5, no replan before the hit 0

## Per scenario

| seed | blocked | other route | B0 | B2 | b2_guided | oracle |
|---|---|---|---|---|---|---|
| 3000 | 5,7 | yes | obstacle_collision | ok | ok | ok |
| 3001 | 3,2 | yes | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 3002 | 4,6 | yes | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 3003 | 1,7 | yes | obstacle_collision | stuck_on_wall | stuck_on_wall | obstacle_collision |
| 3004 | 1,3 | yes | obstacle_collision | ok | obstacle_collision | obstacle_collision |
| 3005 | 4,1 | no | obstacle_collision | obstacle_collision | ok | obstacle_collision |
| 3006 | 6,6 | no | obstacle_collision | obstacle_collision | ok | ok |
| 3007 | 2,4 | yes | obstacle_collision | ok | ok | obstacle_collision |
| 3009 | 4,6 | yes | obstacle_collision | ok | ok | ok |
| 3010 | 3,5 | no | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 3011 | 4,6 | yes | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 3012 | 3,5 | no | obstacle_collision | obstacle_collision | obstacle_collision | obstacle_collision |
| 3013 | 6,2 | no | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 3014 | 3,3 | yes | obstacle_collision | ok | ok | ok |
| 3015 | 6,6 | no | obstacle_collision | obstacle_collision | stuck_on_wall | obstacle_collision |
