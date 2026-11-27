"""
F7 of experiment_progress/走廊偏好實驗計畫_2026-11-27.md (section 8): the training distribution under the planning
endpoints.

For the 24 S scenarios, every 384-step training window cut as SequenceDataset.indices cuts them (no padding for maze2d)
whose step 0 is in cell A and step 383 is in cell G. x7 = share of those windows that never enter the blocked cell b
(each window weighs the same, as in the training loss). A cluster = a run of consecutive matching starts in one path.

    bash experiments/common/setup/run.sh python experiments/blocked/diag/corridor_window.py
Writes results/diag/corridor/window.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner  # noqa: E402

H = 384
OUT = os.path.join(REPO, 'experiments/blocked/results/diag/corridor')


def main():
    geo = json.load(open(os.path.join(OUT, 'geometry.json')))
    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu')
    ds = planner.experiment.dataset
    obs, lengths = ds.fields.observations, ds.fields.path_lengths
    idx = ds.indices
    assert (idx[:, 2] - idx[:, 1] == H).all()
    cells = [np.floor(obs[i, :n, :2] + mu.CELL_OFFSET + 0.5).astype(int) for i, n in enumerate(lengths)]
    rows = []
    for g in geo:
        a, goal, b = np.array(g['A']), np.array(g['G']), np.array(g['b'])
        n = avoid = clusters = cl_avoid = 0
        for i, c in enumerate(cells):
            starts = idx[idx[:, 0] == i, 1]
            if not len(starts):
                continue
            in_b = np.concatenate([[0], np.cumsum((c == b).all(1))])
            ok = (c[starts] == a).all(1) & (c[starts + H - 1] == goal).all(1)
            s = starts[ok]
            if not len(s):
                continue
            av = in_b[s + H] - in_b[s] == 0
            n += len(s)
            avoid += int(av.sum())
            for run in np.split(np.arange(len(s)), np.where(np.diff(s) != 1)[0] + 1):
                clusters += 1
                cl_avoid += float(av[run].mean())
        rows.append(dict(id=g['id'], group=g['group'], y=g['y'], n=n, avoid=avoid, x7=avoid / n if n else None,
                         clusters=clusters, x7_cluster=cl_avoid / clusters if clusters else None))
        print(f"[F7] {g['id']:7s} windows {n:6d} ({clusters:4d} clusters) | avoid b {avoid:6d} x7 "
              f"{(avoid / n if n else float('nan')):.2f} (by cluster {(cl_avoid / clusters if clusters else float('nan')):.2f}) "
              f"| y {g['y']:.2f}", flush=True)
    with open(os.path.join(OUT, 'window.json'), 'w') as f:
        json.dump(dict(H=H, rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
