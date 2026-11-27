"""
Descriptive part of F6 (experiment_progress/走廊偏好實驗計畫_2026-11-27.md, section 7): do the plans that pass the
blocked cell spend their spare time on a turn-back (out into a side corridor or dead end and straight back)?

Reads the final plans saved by corridor_trace.py --mode trace --save_dir (x0-hat at the last step = the plan; the
.npz files are not in the repo), steps 1..382. Turn-back depth d: the cell sequence goes out d cells and straight back.

    bash experiments/common/setup/run.sh python experiments/blocked/diag/corridor_turnback.py --npz ../.vm/out/corridor_npz
Writes results/diag/corridor/turnback.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from experiments.common import maze_utils as mu  # noqa: E402
from corridor_dataset import dedup, turn_back  # noqa: E402

OUT = os.path.join(REPO, 'experiments/blocked/results/diag/corridor')


def depth(seq):
    d = 0
    while turn_back(seq, d + 1):
        d += 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    args = ap.parse_args()
    rows = []
    for g in json.load(open(os.path.join(OUT, 'geometry.json'))):
        f = os.path.join(args.npz, f"{g['id']}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        plans = d['xy'][:, list(d['steps']).index(255), 1:-1].astype(np.float64)
        b, first_d = tuple(g['b']), tuple(g['route_d'][1])
        via, avoid = [], []
        for p in plans:
            seq = dedup([mu.xy_to_cell(q) for q in p])
            rec = dict(depth=depth(seq), along_detour=first_d in seq, cells=len(seq))
            (via if b in seq else avoid).append(rec)
        rows.append(dict(id=g['id'], group=g['group'], n_via=len(via), n_avoid=len(avoid),
                         via_turn_back=sum(r['depth'] >= 1 for r in via), via_turn_back2=sum(r['depth'] >= 2 for r in via),
                         via_depth_median=float(np.median([r['depth'] for r in via])) if via else None,
                         via_along_detour=sum(r['along_detour'] for r in via),
                         avoid_turn_back2=sum(r['depth'] >= 2 for r in avoid)))
        r = rows[-1]
        print(f"[turnback] {g['id']:7s} via b {r['n_via']:2d}: turn-back {r['via_turn_back']:2d} (depth >= 2: {r['via_turn_back2']:2d}, "
              f"median depth {r['via_depth_median']}) start along detour {r['via_along_detour']:2d} | avoid {r['n_avoid']:2d}", flush=True)
    with open(os.path.join(OUT, 'turnback.json'), 'w') as f:
        json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
