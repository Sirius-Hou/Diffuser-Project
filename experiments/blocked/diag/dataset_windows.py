"""
What the frozen model was trained on: every H-step window of the maze2d-large training set that the
Diffuser's SequenceDataset samples (episodes split where the goal changes, no padding).

For each window: the distance the ball travels (arc length; a cell is 1 wide, so this is in cells), the BFS
distance between its first and last cell, and the excess (arc length - BFS cells: how much longer than the
shortest route). Answers: can a 384-step plan in the training distribution cover a 15-21 cell detour at all,
and how often do training windows take a route much longer than the shortest one?

    bash experiments/common/setup/run.sh python experiments/blocked/diag/dataset_windows.py
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env  # noqa: E402


def bfs_table(maze):
    """BFS distance (cells) between every pair of free cells, as a dense array indexed by cell coordinates."""
    W, H = maze.shape
    D = np.full((W, H, W, H), -1, dtype=np.int32)
    for c in mu.free_cells(maze):
        for d in mu.free_cells(maze):
            D[c + d] = len(mu.bfs_path(maze, c, d)) - 1
    return D


def episode_windows(xy, h, stride, D):
    """(arc length, BFS cells) of every h-step window starting every `stride` steps."""
    cells = np.floor(xy + mu.CELL_OFFSET + 0.5).astype(int)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = np.arange(0, len(xy) - h + 1, stride)
    e = s + h - 1
    return np.stack([cum[e] - cum[s], D[cells[s, 0], cells[s, 1], cells[e, 0], cells[e, 1]]], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--horizons', type=int, nargs='*', default=[384, 512, 640])
    ap.add_argument('--stride', type=int, default=25, help='window start every `stride` steps')
    ap.add_argument('--out', default='experiments/blocked/results/diag/dataset_windows')
    args = ap.parse_args()
    out = os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu')
    ds = planner.experiment.dataset
    maze = load_env('maze2d-large-v1').maze_arr
    obs, lengths = ds.fields.observations, ds.fields.path_lengths
    steps = np.linalg.norm(np.diff(obs[:, :, :2], axis=1), axis=-1)
    summary = dict(n_episodes=int(len(lengths)), episode_length=dict(
        median=float(np.median(lengths)), p90=float(np.percentile(lengths, 90)), max=int(lengths.max())),
        step_length_median=float(np.median(np.concatenate([steps[i, :lengths[i] - 1] for i in range(len(lengths))]))))
    D = bfs_table(maze)
    for h in args.horizons:
        parts = [episode_windows(obs[i, :n, :2], h, args.stride, D) for i, n in enumerate(lengths) if n >= h]
        a = np.concatenate(parts) if parts else np.zeros((0, 2))
        path, bfs = a[:, 0], a[:, 1]
        excess = path - bfs
        summary[f'H{h}'] = dict(
            n_windows=int(len(a)),
            episodes_long_enough=int((lengths >= h).sum()),
            path_length=dict(zip(['p10', 'p50', 'p90', 'p99', 'max'],
                                [float(x) for x in np.percentile(path, [10, 50, 90, 99, 100])])) if len(a) else {},
            frac_path_ge={k: float((path >= k).mean()) for k in (12, 15, 18, 21)} if len(a) else {},
            frac_excess_ge={k: float((excess >= k).mean()) for k in (2, 4, 6, 8, 12)} if len(a) else {},
            frac_excess_ge_given_bfs_le_9={k: float((excess[bfs <= 9] >= k).mean()) for k in (4, 8)}
            if (bfs <= 9).any() else {})
        np.save(os.path.join(out, f'windows_H{h}.npy'), a)
        print(f'[data] H={h}: {len(a)} windows | path length {summary[f"H{h}"]["path_length"]} | '
              f'path>=15 {summary[f"H{h}"]["frac_path_ge"].get(15)} | excess>=8 {summary[f"H{h}"]["frac_excess_ge"].get(8)}',
              flush=True)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
