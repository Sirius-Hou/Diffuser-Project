# Denoising probe: summary (exploratory)

Rules: experiment_progress/去噪探針實驗計畫_2026-11-26.md, section 4. "High" = above the 90th percentile of the training windows (DATA p90 = 0.0046).

Excluded physical trajectories (missed G by > 0.5 or points in a wall): 0 

| set | key | n | median score | high (> DATA p90) | median arc | median speed per step |
|---|---|---|---|---|---|---|
| DATA | all | 200 | 0.0033 | 20/200 | 13.2 | 0.0354 |
| SHORT | 5 | 6 | 0.0229 | 6/6 | 5.1 | 0.0140 |
| SHORT | 7 | 6 | 0.0168 | 6/6 | 7.2 | 0.0195 |
| SHORT | 9 | 6 | 0.0093 | 6/6 | 9.3 | 0.0254 |
| SHORT | 11 | 6 | 0.0071 | 5/6 | 11.5 | 0.0309 |
| SHORT | 13 | 6 | 0.0047 | 3/6 | 13.8 | 0.0368 |
| SHORT | 15 | 6 | 0.0097 | 6/6 | 15.9 | 0.0432 |
| SHORT | 17 | 6 | 0.0190 | 6/6 | 17.2 | 0.0509 |
| SHORT | 19 | 6 | 0.0235 | 6/6 | 17.0 | 0.0523 |
| DETOUR | 9 | 6 | 0.0131 | 6/6 | 9.3 | 0.0253 |
| DETOUR | 11 | 6 | 0.0074 | 6/6 | 11.5 | 0.0310 |
| DETOUR | 13 | 6 | 0.0047 | 4/6 | 13.8 | 0.0375 |
| DETOUR | 15 | 6 | 0.0098 | 6/6 | 16.1 | 0.0442 |
| DETOUR | 17 | 6 | 0.0256 | 6/6 | 17.1 | 0.0506 |
| DETOUR | 19 | 6 | 0.0314 | 6/6 | 17.0 | 0.0523 |
| DETOUR | 21 | 6 | 0.0449 | 6/6 | 17.5 | 0.0511 |
| KROUTE | 11 | 6 | 0.0137 | 6/6 | 11.0 | 0.0306 |
| KROUTE | 13 | 6 | 0.0142 | 6/6 | 13.0 | 0.0362 |
| KROUTE | 15 | 6 | 0.0233 | 6/6 | 15.0 | 0.0418 |
| WALL | 11 | 6 | 0.0240 | 6/6 | 8.2 | 0.0228 |
| WALL | 13 | 6 | 0.0304 | 6/6 | 9.1 | 0.0254 |
| WALL | 15 | 6 | 0.0390 | 6/6 | 7.9 | 0.0221 |

C1 (SHORT 11-15 at most 1/3 high): 14/18 high -> FAIL
C2 (WALL >= 80% high and KROUTE at most 1/3 high): WALL 18/18, KROUTE 18/18 -> FAIL

Length / speed: SHORT 17-21 high 12/12, SHORT 11-15 high 14/18
**Length / speed decision: probe not usable (calibration failed).**

Shape: share of same-length (detour, shortest) pairs, T in 9-15, where the detour scores higher: A = 0.68 (98/144)
**Shape decision: probe not usable (calibration failed).**

## Per noise level (median score)

| set | t=8 | t=32 | t=64 | t=128 | t=192 |
|---|---|---|---|---|---|
| DATA | 0.0012 | 0.0017 | 0.0025 | 0.0038 | 0.0067 |
| SHORT | 0.0018 | 0.0034 | 0.0066 | 0.0143 | 0.0352 |
| DETOUR | 0.0019 | 0.0035 | 0.0072 | 0.0144 | 0.0404 |
| KROUTE | 0.0104 | 0.0138 | 0.0121 | 0.0161 | 0.0290 |
| WALL | 0.0041 | 0.0077 | 0.0146 | 0.0474 | 0.0684 |
