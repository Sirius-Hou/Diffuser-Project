"""
Interior inpainting (proposal option b, contribution 3), offline at the moment the obstacle appears: one plan to
the goal with the x-y of some plan points fixed on the A* detour. Does the frozen Diffuser follow them onto
another corridor?

Same states, seeds, scoring and labels as guided_sampling.py (E1); plan file: final_project/補強實驗計畫_2026-11-23.md.

  via      : 'junctions' or 'corners' = maze_graph.waypoints(..., max_leg=4) on the A* route from the cell at
             t_appear with the blocked cell removed (cell centres, goal excluded)
  index    : waypoint at route cell i is fixed at plan index round(R * i / L), R = round(0.94 * (H - 1)), L = route cells
  horizon  : 'fixed' = 384; 'matched' = 4 * ceil(max(384, 28 L / 0.94) / 4), about the training data's 27 steps per cell
  via_passed : per plan, how many via points have a plan point within 0.5 more than 3 steps away from their own
               index (a plan that jumps to a via point and back does not pass it)

    bash experiments/common/setup/run.sh python experiments/blocked/diag/interior_sampling.py --tag I1 --via junctions
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.common import guidance as gd  # noqa: E402
from experiments.common import maze_graph as mg  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds  # noqa: E402
from guided_sampling import label, scenario_meta, scenarios  # noqa: E402
from trigger_sampling import state_at_appear  # noqa: E402

H = 384
STEPS_PER_CELL = 28


def via_points(maze, sc, mode, horizon_mode):
    start, goal, blocked = tuple(sc['cell_at_appear']), tuple(sc['goal_cell']), tuple(sc['obstacle']['cell'])
    path = mg.astar(maze, start, goal, [blocked])
    L = len(path) - 1
    h = H if horizon_mode == 'fixed' else 4 * int(np.ceil(max(H, STEPS_PER_CELL * L / 0.94) / 4))
    R = int(round(0.94 * (h - 1)))
    idx = mg.waypoints(maze, path, [blocked], 4, corners=mode == 'corners')[:-1]
    return h, {int(round(R * i / L)): mu.cell_center(path[i]) for i in idx}, L


def via_passed(plan, via):
    xy = plan[:, :2]
    k = np.arange(len(xy))
    return sum(bool((np.linalg.norm(xy[np.abs(k - j) > 3] - p, axis=1) < 0.5).any()) for j, p in via.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--via', choices=['junctions', 'corners'], default='corners')
    ap.add_argument('--horizon', choices=['fixed', 'matched'], default='fixed')
    ap.add_argument('--scale', type=float, default=0.0)
    ap.add_argument('--guide_margin', type=float, default=0.15)
    ap.add_argument('--walls', type=float, default=0.0)
    ap.add_argument('--wall_margin', type=float, default=0.0)
    ap.add_argument('--smooth', type=float, default=0.0)
    ap.add_argument('--min_extra', type=int, default=0, help='only scenarios whose detour is at least this many cells longer')
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--out', default='experiments/blocked/results/diag/interior_sampling')
    args = ap.parse_args()

    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu', threads=args.threads)
    env = load_env('maze2d-large-v1')
    maze = env.maze_arr
    out = os.path.join(REPO, args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != 'resume'}
    rows = []
    if args.resume and os.path.exists(os.path.join(out, 'rows.json')):
        rows = json.load(open(os.path.join(out, 'rows.json')))['rows']
    done = {(r['results'], r['seed']) for r in rows}
    for name, sc, plan0 in scenarios('full', ['probe', 'probe_multigoal']):
        seed = sc['seed']
        extra = sc['detour_cells'] - sc['base_cells']
        if (args.seeds is not None and seed not in args.seeds) or (name, seed) in done or extra < args.min_extra:
            continue
        state = state_at_appear(env, sc, plan0)
        target = np.array(sc.get('target', env._target), dtype=np.float64)
        obstacle = ob.Obstacle.from_dict(sc['obstacle'])
        cost = gd.Cost(obstacle, args.guide_margin, maze if args.walls else None, wall_weight=args.walls,
                       wall_margin=args.wall_margin, smooth_weight=args.smooth)
        h, via, L = via_points(maze, sc, args.via, args.horizon)
        detour_only, axis = scenario_meta(maze, sc)
        set_all_seeds(700000 + 1000 * seed + H)                  # the trigger_sampling.py / guided_sampling.py rule
        labels, keep, dt = [], [], 0.0
        for _ in range(0, args.n, args.batch):
            t0 = time.perf_counter()
            plans, _, _ = gd.guided_sample(planner, state, target, args.batch, cost, args.scale, horizon=h, via=via)
            dt += time.perf_counter() - t0
            for p in plans:
                s = label(maze, p, obstacle, args.margin, detour_only, axis)
                s['via_passed'] = via_passed(p, via)
                labels.append(s)
                keep.append(p.astype(np.float32))
        if args.save:
            np.save(os.path.join(out, f'{name}_seed{seed:03d}.npy'), np.stack(keep))
        n_valid = sum(s['valid'] for s in labels)
        row = dict(results=name, seed=seed, tag=args.tag, n=len(labels), base_cells=sc['base_cells'],
                   detour_cells=sc['detour_cells'], route_cells=L, horizon=h, n_via=len(via),
                   via={str(k): list(map(float, v)) for k, v in via.items()}, plan_time_s=dt,
                   n_clear=sum(s['clear'] for s in labels), n_valid=n_valid,
                   n_valid_via_detour=sum(bool(s['valid'] and s['via_detour']) for s in labels),
                   n_via_detour=sum(bool(s['via_detour']) for s in labels),
                   fail={k: sum(s['fail'] == k for s in labels) for k in ('still_hit', 'split', 'wall')},
                   median_in_wall=float(np.median([s['in_wall'] for s in labels])),
                   median_max_jump=float(np.median([s['max_jump'] for s in labels])),
                   mean_via_passed=float(np.mean([s['via_passed'] for s in labels])),
                   all_via_passed=sum(s['via_passed'] == len(via) for s in labels), samples=labels)
        rows.append(row)
        print(f"[interior {args.tag}] {name:16s} seed {seed:3d} | extra {extra} | H {h} | via {len(via)} | "
              f"valid {n_valid:2d} (via detour {row['n_valid_via_detour']}) | any detour {row['n_via_detour']} | "
              f"fail {row['fail']} | max jump {row['median_max_jump']:.2f} | via passed "
              f"{row['mean_via_passed']:.1f}/{len(via)} (all {row['all_via_passed']}) | {dt:.0f}s", flush=True)
        with open(os.path.join(out, 'rows.json'), 'w') as f:
            json.dump(dict(config=config, rows=rows), f)


if __name__ == '__main__':
    main()
