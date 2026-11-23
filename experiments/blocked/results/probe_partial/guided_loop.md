# probe_partial: B2 vs b2_guided (exploratory)

Paired scenarios: 7 (dropped: [4])

| Method | all (n=7) | another route (n=3) | no other route (n=4) | hit obstacle | planning time (s, mean) |
|---|---|---|---|---|---|
| B0 | 1/7 = 14% (3-51%) | 1/3 = 33% (6-79%) | 0/4 = 0% (0-49%) | 6/7 | 4 |
| B2 | 1/7 = 14% (3-51%) | 1/3 = 33% (6-79%) | 0/4 = 0% (0-49%) | 4/7 | 101 |
| b2_guided | 1/7 = 14% (3-51%) | 1/3 = 33% (6-79%) | 0/4 = 0% (0-49%) | 5/7 | 155 |
| oracle | 7/7 = 100% (65-100%) | 3/3 = 100% (44-100%) | 4/4 = 100% (51-100%) | 0/7 | 0 |

Discordant pairs: only B2 0, only b2_guided 0; exact McNemar two-sided p = 1.0000.

## Where the obstacle collisions came from

- B2: planning 2, execution 2, no replan before the hit 0
- b2_guided: planning 1, execution 4, no replan before the hit 0

## Per scenario

| seed | blocked | other route | B0 | B2 | b2_guided | oracle |
|---|---|---|---|---|---|---|
| 0 | 6,8 | no | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 1 | 5,9 | yes | obstacle_collision | ok | ok | ok |
| 2 | 6,8 | no | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 3 | 6,8 | no | obstacle_collision | stuck_on_wall | obstacle_collision | ok |
| 5 | 3,5 | no | obstacle_collision | obstacle_collision | obstacle_collision | ok |
| 6 | 5,7 | yes | ok | stuck_on_wall | stuck_on_wall | ok |
| 7 | 5,7 | yes | obstacle_collision | obstacle_collision | obstacle_collision | ok |
