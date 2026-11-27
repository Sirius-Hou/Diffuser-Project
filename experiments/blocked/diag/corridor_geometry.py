"""
F1 of experiment_progress/走廊偏好實驗計畫_2026-11-27.md: how close each candidate route is to the straight line A-G.

For the 24 S scenarios of E1 and E1r (total <= 13 cells):
  R_b : BFS A -> b, then BFS b -> G (the shortest route through the blocked cell)
  R_d : BFS A -> G with b blocked (the detour)
  d(R): mean distance to the segment [centre(A), centre(G)] of 200 points equally spaced along R (cells)
  delta = d(R_d) - d(R_b)   (< 0: the detour is closer to the straight line)
Also records y = share of E1/E1r samples that do not pass b.

    bash experiments/common/setup/run.sh python experiments/blocked/diag/corridor_geometry.py
Writes results/diag/corridor/geometry.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'experiments', 'blocked'))
from experiments.common import maze_utils as mu  # noqa: E402
from analyze_budget_closed_loop import MAZE  # noqa: E402

BUDGET = os.path.join(REPO, 'experiments/blocked/results/diag/budget')
OUT = os.path.join(REPO, 'experiments/blocked/results/diag/corridor')
N_POINTS = 200


def s_rows():
    rows = []
    for exp in ('e1', 'e1r'):
        d = os.path.join(BUDGET, exp, 'rows')
        rows += [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith('.json')]
    return [r for r in rows if r['group'].startswith('S-')]


def routes(a, g, b):
    r_b = mu.bfs_path(MAZE, a, b) + mu.bfs_path(MAZE, b, g)[1:]
    r_d = mu.bfs_path(MAZE, a, g, [b])
    return [tuple(map(int, c)) for c in r_b], [tuple(map(int, c)) for c in r_d]


def resample(cells, n=N_POINTS):
    pts = np.array([mu.cell_center(c) for c in cells], dtype=np.float64)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0.0, cum[-1], n)
    return np.c_[np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]


def seg_dist(p, a, g):
    """Distance of each point in p [N, 2] to the segment a-g."""
    v = g - a
    u = np.clip(((p - a) @ v) / (v @ v), 0.0, 1.0)
    return np.linalg.norm(p - (a + u[:, None] * v), axis=1)


def line_dist(cells):
    a, g = mu.cell_center(cells[0]), mu.cell_center(cells[-1])
    return float(seg_dist(resample(cells), a, g).mean())


def main():
    out = []
    for r in s_rows():
        a, g, b = tuple(r['A']), tuple(r['G']), tuple(r['b'])
        r_b, r_d = routes(a, g, b)
        assert len(r_d) - 1 == r['total'] and len(r_b) - 1 == r['base'], r['id']
        d_b, d_d = line_dist(r_b), line_dist(r_d)
        out.append(dict(id=r['id'], group=r['group'], A=list(a), G=list(g), b=list(b), base=r['base'], total=r['total'],
                        extra=r['extra'], route_b=[list(c) for c in r_b], route_d=[list(c) for c in r_d],
                        d_line_b=d_b, d_line_d=d_d, delta=d_d - d_b, y=1 - r['n_via_b'] / r['n'], n=r['n']))
        print(f"[F1] {r['id']:7s} {r['group']:8s} base {r['base']:2d} total {r['total']:2d} | d_b {d_b:.2f} d_d {d_d:.2f} "
              f"delta {d_d - d_b:+.2f} | y {out[-1]['y']:.2f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'geometry.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
