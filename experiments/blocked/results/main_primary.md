# main: pre-registered analysis

Scenarios: 46 paired (dropped: [1005, 1007, 1013, 1039]); 29 with a longer detour, 17 with a same-length route.

## Primary: B2 vs ours (junctions), full blockage with a longer detour

| | n | Success (Wilson 95% CI) |
|---|---|---|
| B2 | 29 | 10/29 = 34% (20-53%) |
| ours (junctions) | 29 | 23/29 = 79% (62-90%) |

Discordant pairs: only B2 succeeds 0, only ours (junctions) succeeds 13; both 10, neither 6. Exact McNemar two-sided p = 0.0002.

## Secondary 1 and 3: every method by case

| Method | Longer detour | Same length | All |
|---|---|---|---|
| B0 | 0/29 = 0% (0-12%) | 7/17 = 41% (22-64%) | 7/46 = 15% (8-28%) |
| B1 | 14/29 = 48% (31-66%) | 11/17 = 65% (41-83%) | 25/46 = 54% (40-68%) |
| B2 | 10/29 = 34% (20-53%) | 15/17 = 88% (66-97%) | 25/46 = 54% (40-68%) |
| ours (corners) | 25/29 = 86% (69-95%) | 16/17 = 94% (73-99%) | 41/46 = 89% (77-95%) |
| ours (junctions) | 23/29 = 79% (62-90%) | 16/17 = 94% (73-99%) | 39/46 = 85% (72-92%) |
| astar_line + lag | 29/29 = 100% (88-100%) | 17/17 = 100% (82-100%) | 46/46 = 100% (92-100%) |
| oracle | 29/29 = 100% (88-100%) | 17/17 = 100% (82-100%) | 46/46 = 100% (92-100%) |

## Secondary 2: success by extra detour cells (longer detours only)

| Method | extra <= 6 (n=18) | extra >= 8 (n=11) |
|---|---|---|
| B0 | 0/18 = 0% (0-18%) | 0/11 = 0% (0-26%) |
| B1 | 10/18 = 56% (34-75%) | 4/11 = 36% (15-65%) |
| B2 | 8/18 = 44% (25-66%) | 2/11 = 18% (5-48%) |
| ours (corners) | 17/18 = 94% (74-99%) | 8/11 = 73% (43-90%) |
| ours (junctions) | 14/18 = 78% (55-91%) | 9/11 = 82% (52-95%) |
| astar_line + lag | 18/18 = 100% (82-100%) | 11/11 = 100% (74-100%) |
| oracle | 18/18 = 100% (82-100%) | 11/11 = 100% (74-100%) |

## Secondary 4: success radius (longer detours; min distance to the goal over the episode)

| Method | r = 0.5 | r = 0.75 | r = 1.0 |
|---|---|---|---|
| B0 | 0/29 = 0% (0-12%) | 0/29 = 0% (0-12%) | 0/29 = 0% (0-12%) |
| B1 | 14/29 = 48% (31-66%) | 16/29 = 55% (38-72%) | 17/29 = 59% (41-74%) |
| B2 | 10/29 = 34% (20-53%) | 10/29 = 34% (20-53%) | 10/29 = 34% (20-53%) |
| ours (corners) | 25/29 = 86% (69-95%) | 25/29 = 86% (69-95%) | 26/29 = 90% (74-96%) |
| ours (junctions) | 23/29 = 79% (62-90%) | 24/29 = 83% (65-92%) | 24/29 = 83% (65-92%) |
| astar_line + lag | 29/29 = 100% (88-100%) | 29/29 = 100% (88-100%) | 29/29 = 100% (88-100%) |
| oracle | 29/29 = 100% (88-100%) | 29/29 = 100% (88-100%) | 29/29 = 100% (88-100%) |

## Secondary 5: A* use and blocked cells (longer detours)

Blocked cells: (4, 6): 6, (3, 3): 5, (5, 7): 4, (3, 2): 3, (1, 7): 2, (1, 6): 2, (1, 8): 1, (5, 9): 1, (3, 6): 1, (5, 8): 1, (5, 6): 1, (1, 4): 1, (2, 4): 1
- ours (corners): A* used in 20/29, success 17/20 in those
- ours (junctions): A* used in 20/29, success 15/20 in those
- astar_line + lag: A* used in 23/29, success 23/23 in those
