"""
F6 of experiment_progress/走廊偏好實驗計畫_2026-11-27.md (section 7): time-matched data frequency.

For the 24 S scenarios, training segments from the last step in A to the first arrival in G (as budget_dataset.py)
that take 300-420 steps (a plan reaches G at step 359). x6 = share of them that avoid the blocked cell b.
Also: share of segments with a turn-back in their cell sequence (c -> d -> c; depth 2: c -> d -> e -> d -> c).

    bash experiments/common/setup/run.sh python experiments/blocked/diag/corridor_dataset.py
Writes results/diag/corridor/dataset.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner  # noqa: E402
from budget_dataset import segments  # noqa: E402

LO, HI = 300, 420
OUT = os.path.join(REPO, 'experiments/blocked/results/diag/corridor')


def turn_back(seq, depth):
    """Cell sequence (no repeats) goes out `depth` cells and straight back somewhere."""
    w = 2 * depth
    return any(all(seq[i + k] == seq[i + w - k] for k in range(depth)) for i in range(len(seq) - w))


def dedup(cells):
    return [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]


def main():
    geo = json.load(open(os.path.join(OUT, 'geometry.json')))
    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu')
    ds = planner.experiment.dataset
    obs, lengths = ds.fields.observations, ds.fields.path_lengths
    cells = [[tuple(int(v) for v in c) for c in np.floor(obs[i, :n, :2] + mu.CELL_OFFSET + 0.5).astype(int)]
             for i, n in enumerate(lengths)]
    rows = []
    for g in geo:
        a, goal, b = tuple(g['A']), tuple(g['G']), tuple(g['b'])
        n = avoid = tb1 = tb2 = tb1_via = tb2_via = 0
        for ep in cells:
            for t0, t1 in segments(ep, a, goal):
                if not LO <= t1 - t0 <= HI:
                    continue
                seg = ep[t0:t1 + 1]
                seq = dedup(seg)
                n += 1
                via = b in seg
                avoid += not via
                t1_, t2_ = turn_back(seq, 1), turn_back(seq, 2)
                tb1 += t1_
                tb2 += t2_
                tb1_via += via and t1_
                tb2_via += via and t2_
        rows.append(dict(id=g['id'], group=g['group'], y=g['y'], n=n, avoid=avoid, x6=avoid / n if n else None,
                         turn_back1=tb1, turn_back2=tb2, via=n - avoid, turn_back1_via=tb1_via, turn_back2_via=tb2_via))
        print(f"[F6] {g['id']:7s} segments {LO}-{HI} steps {n:4d} | avoid b {avoid:4d} x6 "
              f"{(avoid / n if n else float('nan')):.2f} | y {g['y']:.2f} | via b {n - avoid}: turn-back {tb1_via} (depth 2: {tb2_via})",
              flush=True)
    with open(os.path.join(OUT, 'dataset.json'), 'w') as f:
        json.dump(dict(window=[LO, HI], rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
