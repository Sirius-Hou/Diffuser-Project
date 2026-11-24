# Interior inpainting (proposal option b), offline

Valid plans that also visit a detour-only cell, out of 64 per scenario. Decision rule (written before running): a group passes if >= 2 of the 13 scenarios with a detour >= 8 cells longer have >= 4/64.

| Group | Via points | Detour >= 8 longer | Detour <= 6 longer | Scenarios >= 4/64 (>= 8) | Failures (>= 8) | Median max jump (>= 8) | Via points passed (>= 8, mean) |
|---|---|---|---|---|---|---|---|
| I1_junctions_h384 | junctions, H 384 | 0/832 | 30/576 | 0/13 | {'still_hit': 831, 'split': 1, 'wall': 0} | 4.31 | 42% |
| I2_corners_h384 | junctions + corners, H 384 | 0/832 | 30/576 | 0/13 | {'still_hit': 832, 'split': 0, 'wall': 0} | 4.28 | 41% |
| I3_corners_matched | junctions + corners, H by length | 0/832 | 31/576 | 0/13 | {'still_hit': 827, 'split': 5, 'wall': 0} | 4.30 | 42% |
| I4_corners_matched_ws5 | as I3 + guidance (obstacle, walls, smooth) | 0/832 | not run | 0/13 | {'still_hit': 825, 'split': 7, 'wall': 0} | 4.40 | 42% |

G0 (no via points, same states and seeds; guided_sampling): detour >= 8 longer 0/832, <= 6 longer 84/576.

Decision: no group passes: interior inpainting does not move the plan onto the detour.
