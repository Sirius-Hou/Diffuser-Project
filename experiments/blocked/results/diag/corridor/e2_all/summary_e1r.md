# E2: A -> G segments in the training set (descriptive)

1062 training episodes (split where the goal changes); segments of at most 1000 steps.

| group | segments A -> G | avoiding b | avoiding b within 384 steps |
|---|---|---|---|
| S-small | 3000 | 963 | 532 |
| S-large | 6789 | 1052 | 530 |
| L-small | 1904 | 108 | 0 |
| L-large | 0 | 0 | 0 |

| id | group | total | extra | segments | avoiding b | within 384 | median steps (all / avoiding b) |
|---|---|---|---|---|---|---|---|
| e1r_00 | S-small | 8 | 4 | 986 | 181 | 96 | 97.0 / 322.0 |
| e1r_01 | S-small | 13 | 4 | 335 | 24 | 22 | 244.0 / 358.0 |
| e1r_02 | S-small | 12 | 4 | 196 | 93 | 16 | 555.0 / 724.0 |
| e1r_03 | S-small | 12 | 2 | 540 | 236 | 123 | 459.5 / 337.5 |
| e1r_04 | S-small | 12 | 2 | 104 | 24 | 24 | 644.0 / 330.0 |
| e1r_05 | S-small | 10 | 4 | 839 | 405 | 251 | 252.0 / 274.0 |
| e1r_06 | S-large | 13 | 8 | 980 | 169 | 135 | 350.0 / 345.0 |
| e1r_07 | S-large | 13 | 10 | 1939 | 333 | 252 | 328.0 / 346.0 |
| e1r_08 | S-large | 12 | 8 | 456 | 79 | 9 | 90.0 / 769.0 |
| e1r_09 | S-large | 13 | 10 | 2130 | 278 | 112 | 58.0 / 452.5 |
| e1r_10 | S-large | 12 | 8 | 452 | 98 | 12 | 93.0 / 662.0 |
| e1r_11 | S-large | 12 | 8 | 832 | 95 | 10 | 94.0 / 680.0 |
| e1r_12 | L-small | 18 | 4 | 281 | 21 | 0 | 491.0 / 560.0 |
| e1r_13 | L-small | 17 | 4 | 327 | 17 | 0 | 360.0 / 475.0 |
| e1r_14 | L-small | 18 | 4 | 225 | 8 | 0 | 385.0 / 598.0 |
| e1r_15 | L-small | 17 | 4 | 348 | 19 | 0 | 436.5 / 457.0 |
| e1r_16 | L-small | 18 | 4 | 150 | 5 | 0 | 465.0 / 492.0 |
| e1r_17 | L-small | 17 | 4 | 573 | 38 | 0 | 435.0 / 488.0 |
