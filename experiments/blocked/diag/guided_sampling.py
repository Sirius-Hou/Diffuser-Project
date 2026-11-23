"""
Offline guidance at the moment the obstacle appears: can a hand-written obstacle cost steer the frozen
Diffuser onto another corridor (full blockage) or through the gap (partial blockage)?

Same states, seeds and scoring as trigger_sampling.py (diagnostic #2), H = 384 only; the sampler is
experiments/common/guidance.py. Extra per-plan labels:

  fail     : 'still_hit' (a plan point touches the obstacle), 'split' (no point does, but the plan jumps over it:
             a midpoint touches it or max_jump >= 0.5), 'wall' (clear, in_wall > 5), or None (valid)
  via_detour : the plan visits a cell of the A* detour (with the obstacle cell removed) that is not on the
               A* route, so a valid plan really took another corridor
  gap_dev  : partial only, mean |offset| across the corridor from the gap centre over the plan points in
             the obstacle cell (the gap is 0.5 wide; the ball needs about +-0.12 of the centre line)

    bash experiments/common/setup/run.sh python experiments/blocked/diag/guided_sampling.py --check
    bash experiments/common/setup/run.sh python experiments/blocked/diag/guided_sampling.py --tag G3 --scale 0.1
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
from trigger_sampling import score, state_at_appear  # noqa: E402

H = 384


def scenarios(kind, results):
    for name in results:
        base = os.path.join(REPO, 'experiments/blocked/results', name)
        plans0 = np.load(os.path.join(base, 'plans0.npz'))
        for sc in json.load(open(os.path.join(base, 'scenarios.json')))['scenarios']:
            if sc['obstacle']['kind'] != kind:
                continue
            if kind == 'full' and sc['detour_cells'] <= sc['base_cells']:
                continue
            yield name, sc, plans0[f'seed{sc["seed"]:03d}']


def scenario_meta(maze, sc):
    """Cells on the A* detour but not on the A* route (from the cell at t_appear), and the corridor axis."""
    start, goal = tuple(sc['cell_at_appear']), tuple(sc['goal_cell'])
    route = mg.astar(maze, start, goal) or []
    detour = mg.astar(maze, start, goal, blocked=[tuple(sc['obstacle']['cell'])]) or []
    return set(detour) - set(route), ob.corridor_axis(maze, tuple(sc['obstacle']['cell']))


def label(maze, plan, obstacle, margin, detour_only, axis):
    s = score(maze, plan, obstacle, margin)
    s['hit_pts'] = int((obstacle.distance(plan[:, :2]) < mu.BALL_RADIUS + margin).sum())
    s['fail'] = None if s['valid'] else ('still_hit' if s['hit_pts'] else
                                         'split' if not s['clear'] or s['max_jump'] >= 0.5 else 'wall')
    cells = set(mu.cell_sequence(plan[:, :2]))
    s['via_detour'] = bool(cells & detour_only) if detour_only else None
    if obstacle.kind == 'partial' and axis is not None:
        xy = plan[:, :2]
        inside = np.array([mu.xy_to_cell(p) == obstacle.cell for p in xy])
        w = 1 - axis
        s['gap_dev'] = float(np.abs(xy[inside, w] - obstacle.gap_center()[w]).mean()) if inside.any() else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=['full', 'partial'], default='full')
    ap.add_argument('--results', nargs='*', default=None,
                    help='default: probe probe_multigoal (full) or probe_partial (partial)')
    ap.add_argument('--tag', default='G0')
    ap.add_argument('--scale', type=float, default=0.0)
    ap.add_argument('--guide_margin', type=float, default=0.15)
    ap.add_argument('--window', choices=gd.WINDOWS, default='all')
    ap.add_argument('--variant', choices=['x0', 'xt'], default='x0')
    ap.add_argument('--walls', type=float, default=0.0, help='weight of the wall cost (0 = obstacle only)')
    ap.add_argument('--wall_margin', type=float, default=0.0)
    ap.add_argument('--smooth', type=float, default=0.0, help='weight of the step-length cost (0 = off)')
    ap.add_argument('--max_step', type=float, default=0.15)
    ap.add_argument('--seeds', type=int, nargs='*', default=None, help='only these scenario seeds')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--margin', type=float, default=0.05, help='check_margin of the probe configs')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--resume', action='store_true', help='keep the rows already in <tag>/rows.json, skip those scenarios')
    ap.add_argument('--save', action='store_true', help='also save the plans (.npy, not committed)')
    ap.add_argument('--check', action='store_true',
                    help='E0: scale 0 must equal planner.plan; then time one guided batch')
    ap.add_argument('--out', default='experiments/blocked/results/diag/guided_sampling')
    args = ap.parse_args()
    results = args.results or (['probe', 'probe_multigoal'] if args.kind == 'full' else ['probe_partial'])

    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu', threads=args.threads)
    env = load_env('maze2d-large-v1')
    maze = env.maze_arr
    walls = maze

    if args.check:
        name, sc, plan0 = next(scenarios('full', ['probe']))
        state = state_at_appear(env, sc, plan0)
        target = np.array(sc.get('target', env._target), dtype=np.float64)
        obstacle = ob.Obstacle.from_dict(sc['obstacle'])
        set_all_seeds(1)
        ref, _, t_ref = planner.plan(state, target, n=args.batch)
        set_all_seeds(1)
        new, _, info = gd.guided_sample(planner, state, target, args.batch)
        print(f'[check] scale 0 vs planner.plan: max |diff| = {np.abs(ref - new).max():.3e} '
              f'(allclose {np.allclose(ref, new, atol=1e-5)}), plan {t_ref:.1f}s', flush=True)
        for variant, scale in (('x0', 0.1), ('xt', 0.1)):
            for walls_w in (0.0, 1.0):
                set_all_seeds(1)
                cost = gd.Cost(obstacle, args.guide_margin, walls if walls_w else None, wall_weight=walls_w)
                t0 = time.perf_counter()
                p, _, info = gd.guided_sample(planner, state, target, args.batch, cost, scale, variant=variant)
                dt = time.perf_counter() - t0
                sc_ = [score(maze, q, obstacle, args.margin) for q in p]
                d_ref = obstacle.distance(ref[:, :, :2]).min(axis=1)
                d_new = obstacle.distance(p[:, :, :2]).min(axis=1)
                print(f'[check] {variant} scale {scale} walls {walls_w}: {dt:.1f}s / {args.batch}, nan {np.isnan(p).any()}, '
                      f'push {info["push_mean"]:.4f}, min dist to obstacle {np.median(d_ref):.3f} -> '
                      f'{np.median(d_new):.3f} (median), clear {sum(s["clear"] for s in sc_)}, '
                      f'valid {sum(s["valid"] for s in sc_)}, in_wall median {np.median([s["in_wall"] for s in sc_]):.0f}',
                      flush=True)
                np.save(os.path.join(REPO, args.out, f'check_{variant}_w{walls_w:g}.npy'), p.astype(np.float32))
        np.save(os.path.join(REPO, args.out, 'check_ref.npy'), ref.astype(np.float32))
        return

    out = os.path.join(REPO, args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('check', 'resume')}
    rows = []
    if args.resume and os.path.exists(os.path.join(out, 'rows.json')):
        rows = json.load(open(os.path.join(out, 'rows.json')))['rows']   # seeds are per scenario, so this is exact
    done = {(r['results'], r['seed']) for r in rows}
    for name, sc, plan0 in scenarios(args.kind, results):
        seed = sc['seed']
        if (args.seeds is not None and seed not in args.seeds) or (name, seed) in done:
            continue
        state = state_at_appear(env, sc, plan0)
        target = np.array(sc.get('target', env._target), dtype=np.float64)
        obstacle = ob.Obstacle.from_dict(sc['obstacle'])
        cost = gd.Cost(obstacle, args.guide_margin, walls if args.walls else None, wall_weight=args.walls,
                       wall_margin=args.wall_margin, smooth_weight=args.smooth, max_step=args.max_step)
        detour_only, axis = scenario_meta(maze, sc)
        set_all_seeds(700000 + 1000 * seed + H)                  # the trigger_sampling.py rule
        labels, keep, dt, push = [], [], 0.0, []
        for _ in range(0, args.n, args.batch):
            t0 = time.perf_counter()
            plans, _, info = gd.guided_sample(planner, state, target, args.batch, cost, args.scale,
                                              args.window, args.variant)
            dt += time.perf_counter() - t0
            push.append(info['push_mean'])
            for p in plans:
                labels.append(label(maze, p, obstacle, args.margin, detour_only, axis))
                keep.append(p.astype(np.float32))
        if args.save:
            np.save(os.path.join(out, f'{name}_seed{seed:03d}.npy'), np.stack(keep))
        n_valid = sum(s['valid'] for s in labels)
        gaps = [s['gap_dev'] for s in labels if s['valid'] and s.get('gap_dev') is not None]
        row = dict(results=name, seed=seed, tag=args.tag, n=len(labels), kind=sc['obstacle']['kind'],
                   base_cells=sc['base_cells'], detour_cells=sc['detour_cells'], blocked=sc['obstacle']['cell'],
                   plan_time_s=dt, push_mean=float(np.mean(push)),
                   n_clear=sum(s['clear'] for s in labels), n_valid=n_valid,
                   n_valid_via_detour=sum(bool(s['valid'] and s['via_detour']) for s in labels),
                   fail={k: sum(s['fail'] == k for s in labels) for k in ('still_hit', 'split', 'wall')},
                   median_in_wall=float(np.median([s['in_wall'] for s in labels])),
                   median_hit_pts=float(np.median([s['hit_pts'] for s in labels])),
                   median_gap_dev_valid=float(np.median(gaps)) if gaps else None,
                   samples=labels)
        rows.append(row)
        extra = 'none' if sc['detour_cells'] is None else sc['detour_cells'] - sc['base_cells']
        print(f"[guide {args.tag}] {name:16s} seed {seed:3d} | extra {extra} | clear {row['n_clear']:2d} "
              f"valid {n_valid:2d} (via detour {row['n_valid_via_detour']}) | fail {row['fail']} | in-wall "
              f"{row['median_in_wall']:.0f} | hit pts {row['median_hit_pts']:.0f} | push {row['push_mean']:.4f} | {dt:.0f}s", flush=True)
        with open(os.path.join(out, 'rows.json'), 'w') as f:
            json.dump(dict(config=config, rows=rows), f)


if __name__ == '__main__':
    main()
