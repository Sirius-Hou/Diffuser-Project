"""
Offline best-of-N at the moment the obstacle appears: which detours does the frozen Diffuser sample at all?

For every scenario whose detour is longer than its route, the ball is replayed from the stored start state
along the stored t=0 plan (as every method does before t_appear) to the state at t_appear. From there we
sample n plans to the goal for each horizon (384 = trained; longer ones are outside training) and score each:

  clear    : no point touches the obstacle (the B2 filter)
  in_wall  : plan points inside a wall cell;  max_jump : largest step between consecutive points
  valid    : clear, in_wall <= 5 and max_jump < 0.5 (a plan the ball could follow)
  length   : arc length in cells

The obstacle is not an input of the model, so `clear` is simply how often the model goes another way.
With --rest the plans are conditioned on the position only (velocity 0), as ours conditions its legs: does
the ball's momentum towards the obstacle keep the plans on the blocked route?

    bash experiments/common/setup/run.sh python experiments/blocked/diag/trigger_sampling.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds, waypoint_action  # noqa: E402


def state_at_appear(env, sc, plan0):
    env.seed(sc['seed'])
    env.reset()
    env.set_state(np.array(sc['qpos']), np.array(sc['qvel']))
    for t in range(sc['t_appear']):
        env.step(waypoint_action(plan0, t, env.state_vector().copy()))
    return env.state_vector().copy()


def score(maze, plan, obstacle, margin):
    xy = plan[:, :2]
    in_wall = mu.count_points_in_wall(maze, xy)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    clear = ob.first_collision(xy, obstacle, margin) is None
    return dict(clear=bool(clear), in_wall=int(in_wall), max_jump=float(steps.max()), length=float(steps.sum()),
                valid=bool(clear and in_wall <= 5 and steps.max() < 0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', nargs='*', default=['probe', 'probe_multigoal'])
    ap.add_argument('--horizons', type=int, nargs='*', default=[384, 512, 640])
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--margin', type=float, default=0.05, help='check_margin of the probe configs')
    ap.add_argument('--rest', action='store_true', help='condition on the position only')
    ap.add_argument('--out', default='experiments/blocked/results/diag/trigger_sampling')
    args = ap.parse_args()
    out = os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu')
    env = load_env('maze2d-large-v1')
    maze = env.maze_arr
    rows = []
    for name in args.results:
        base = os.path.join(REPO, 'experiments/blocked/results', name)
        scen = json.load(open(os.path.join(base, 'scenarios.json')))['scenarios']
        plans0 = np.load(os.path.join(base, 'plans0.npz'))
        for sc in scen:
            if sc['obstacle']['kind'] != 'full' or sc['detour_cells'] <= sc['base_cells']:
                continue
            seed = sc['seed']
            state = state_at_appear(env, sc, plans0[f'seed{seed:03d}'])
            speed = float(np.linalg.norm(state[2:]))
            if args.rest:
                state = np.r_[state[:2], 0.0, 0.0]
            target = np.array(sc.get('target', env._target), dtype=np.float64)
            obstacle = ob.Obstacle.from_dict(sc['obstacle'])
            for h in args.horizons:
                set_all_seeds(700000 + 1000 * seed + h)
                scores, dt, keep = [], 0.0, []
                for _ in range(0, args.n, args.batch):
                    plans, _, d = planner.plan(state, target, n=args.batch, horizon=h, allow_longer=True)
                    dt += d
                    for p in plans:
                        scores.append(score(maze, p, obstacle, args.margin))
                        keep.append(p.astype(np.float32))
                np.save(os.path.join(out, f'{name}_seed{seed:03d}_H{h}.npy'), np.stack(keep))
                row = dict(results=name, seed=seed, horizon=h, n=len(scores), t_appear=sc['t_appear'],
                           cell_at_appear=sc['cell_at_appear'], base_cells=sc['base_cells'],
                           detour_cells=sc['detour_cells'], blocked=sc['obstacle']['cell'], plan_time_s=dt,
                           speed_at_appear=speed, rest=args.rest,
                           n_clear=sum(s['clear'] for s in scores), n_valid=sum(s['valid'] for s in scores),
                           median_in_wall=float(np.median([s['in_wall'] for s in scores])),
                           median_length=float(np.median([s['length'] for s in scores])),
                           median_length_valid=float(np.median([s['length'] for s in scores if s['valid']]))
                           if any(s['valid'] for s in scores) else None,
                           samples=scores)
                rows.append(row)
                print(f"[trig] {name:16s} seed {seed:3d} | route {sc['base_cells']:2d} -> detour {sc['detour_cells']:2d} "
                      f"| H={h} | clear {row['n_clear']:2d}/{row['n']} valid {row['n_valid']:2d} | in-wall median "
                      f"{row['median_in_wall']:.0f} | length {row['median_length']:.1f} | {dt:.0f}s", flush=True)
                with open(os.path.join(out, 'rows.json'), 'w') as f:
                    json.dump(rows, f)


if __name__ == '__main__':
    main()
