# Guidance diagnostic (offline, H = 384, 64 plans per scenario)

## E1: full blockage (22 scenarios with a longer detour)

| Group | scale | cost | extra <= 6: valid | extra >= 8: valid (via detour) | >= 8 scenarios with >= 4 valid+via | >= 8 fail: still_hit / split / wall | median in-wall |
|---|---|---|---|---|---|---|---|
| full_G0 | 0.0 | obstacle | 85/576 | 0/832 (0) | 0/13 | 830 / 0 / 2 | 0 |
| full_G1_x0_s0.3 | 0.3 | obstacle (x0) | 85/576 | 0/832 (0) | 0/13 | 828 / 1 / 3 | 0 |
| full_G2_x0_s1 | 1.0 | obstacle (x0) | 89/576 | 0/832 (0) | 0/13 | 0 / 827 / 5 | 9 |
| full_G3_x0_s3 | 3.0 | obstacle (x0) | 89/576 | 0/832 (0) | 0/13 | 0 / 826 / 6 | 9 |
| full_G4_ws5_s1 | 1.0 | walls+smooth (x0) | 75/576 | 0/832 (0) | 0/13 | 827 / 1 / 4 | 2 |
| full_G6_xt_s300 | 300.0 | obstacle (xt) | 12/576 | 0/832 (0) | 0/13 | 790 / 0 / 42 | 25 |

## E2: partial blockage, gap 0.5 (probe_partial, 8 scenarios)

| Group | scale | cost | other route: valid | no other route: valid | no-route scenarios above P0 | median gap offset of valid plans |
|---|---|---|---|---|---|---|
| partial_P0 | 0.0 | obstacle | 41/192 | 4/320 (1%) | -/5 | 0.067 |
| partial_P2_x0_s1 | 1.0 | obstacle | 47/192 | 85/320 (27%) | 4/5 | 0.016 |
| partial_P3_x0_s3 | 3.0 | obstacle | 47/192 | 92/320 (29%) | 4/5 | 0.000 |
| partial_P4_ws5_s1 | 1.0 | walls+smooth | 47/192 | 101/320 (32%) | 5/5 | 0.018 |

## Decision (section 10)

- A (another corridor): groups with >= 2 long-detour scenarios having >= 4/64 valid plans through the detour: none
- C (short detours easier): groups with extra <= 6 valid >= 2 x G0 (170): none
- => no closed loop; negative result
