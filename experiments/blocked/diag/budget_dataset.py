"""
E2 of experiment_progress/時間預算實驗計畫_2026-11-25.md (descriptive only): does the training set contain the
routes E1 asks for?

For every E1 scenario (A, G, b), in every training episode as the Diffuser's SequenceDataset splits them (where the
goal changes), find each segment from the last step in A to the next first arrival in G (at most 1000 steps).
Records how many there are, how many avoid the blocked cell b, how long those take, and how many fit in 384 steps.

    bash experiments/common/setup/run.sh python experiments/blocked/diag/budget_dataset.py
    bash experiments/common/setup/run.sh python experiments/blocked/diag/budget_dataset.py --exp e1r --out ../corridor/e2_all
Writes results/diag/budget/e2/summary.{md,json} (--out: a directory relative to results/diag/budget).
--exp e1r was added for F2 of experiment_progress/走廊偏好實驗計畫_2026-11-27.md.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner  # noqa: E402

MAX_STEPS = 1000
H = 384


def segments(cells, a, g):
    """(start, end) of every segment from the last step in cell a to the next first arrival in cell g."""
    out, last_a = [], None
    for t, c in enumerate(cells):
        if c == a:
            last_a = t
        elif c == g and last_a is not None:
            if t - last_a <= MAX_STEPS:
                out.append((last_a, t))
            last_a = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=['e1', 'e1r'], default='e1')
    ap.add_argument('--out', default='e2')
    args = ap.parse_args()
    base = os.path.join(REPO, 'experiments/blocked/results/diag/budget')
    scen = json.load(open(os.path.join(base, args.exp, 'scenarios.json')))['scenarios']
    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu')
    ds = planner.experiment.dataset
    obs, lengths = ds.fields.observations, ds.fields.path_lengths
    cells = [[tuple(int(v) for v in c) for c in np.floor(obs[i, :n, :2] + mu.CELL_OFFSET + 0.5).astype(int)]
             for i, n in enumerate(lengths)]
    rows = []
    for s in scen:
        a, g, b = tuple(s['A']), tuple(s['G']), tuple(s['b'])
        seg, avoid = [], []
        for ep in cells:
            for t0, t1 in segments(ep, a, g):
                seg.append(t1 - t0)
                if b not in ep[t0:t1 + 1]:
                    avoid.append(t1 - t0)
        rows.append(dict(id=s['id'], group=s['group'], total=s['total'], extra=s['extra'], n_segments=len(seg),
                         n_avoid_b=len(avoid), avoid_steps_median=float(np.median(avoid)) if avoid else None,
                         avoid_le_H=sum(x <= H for x in avoid), all_le_H=sum(x <= H for x in seg),
                         all_steps_median=float(np.median(seg)) if seg else None))
        print(f"[e2] {s['id']} {s['group']:8s} total {s['total']:2d} extra {s['extra']:2d} | segments {len(seg):4d} | "
              f"avoid b {len(avoid):3d} (<= {H} steps: {rows[-1]['avoid_le_H']}) | median steps all "
              f"{rows[-1]['all_steps_median']} avoid {rows[-1]['avoid_steps_median']}", flush=True)
    groups = {}
    for gname in ('S-small', 'S-large', 'L-small', 'L-large'):
        rs = [r for r in rows if r['group'] == gname]
        groups[gname] = dict(n_segments=sum(r['n_segments'] for r in rs), n_avoid_b=sum(r['n_avoid_b'] for r in rs),
                             avoid_le_H=sum(r['avoid_le_H'] for r in rs))
    out = os.path.normpath(os.path.join(base, args.out))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'summary_{args.exp}.json' if args.exp != 'e1' or args.out != 'e2' else 'summary.json'), 'w') as f:
        json.dump(dict(n_episodes=int(len(lengths)), groups=groups, rows=rows), f, indent=2)
    lines = ['# E2: A -> G segments in the training set (descriptive)', '',
             f'{len(lengths)} training episodes (split where the goal changes); segments of at most {MAX_STEPS} steps.', '',
             '| group | segments A -> G | avoiding b | avoiding b within 384 steps |', '|---|---|---|---|']
    lines += [f"| {g} | {p['n_segments']} | {p['n_avoid_b']} | {p['avoid_le_H']} |" for g, p in groups.items()]
    lines += ['', '| id | group | total | extra | segments | avoiding b | within 384 | median steps (all / avoiding b) |',
              '|---|---|---|---|---|---|---|---|']
    lines += [f"| {r['id']} | {r['group']} | {r['total']} | {r['extra']} | {r['n_segments']} | {r['n_avoid_b']} | "
              f"{r['avoid_le_H']} | {r['all_steps_median']} / {r['avoid_steps_median']} |" for r in rows]
    with open(os.path.join(out, f'summary_{args.exp}.md' if args.exp != 'e1' or args.out != 'e2' else 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
