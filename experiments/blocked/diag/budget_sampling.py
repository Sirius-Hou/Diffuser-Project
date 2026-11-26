"""
Time budget or route rarity: why does the frozen Diffuser never sample a long detour?
(experiment_progress/時間預算實驗計畫_2026-11-25.md, E0 and E1; exploratory, decision rules fixed in the plan.)

The ball is at rest at the centre of a start cell A, the goal is the centre of cell G, H = 384, 64 samples each.

  e0 : no obstacle; start cells far from the d4rl goal (14-18 cells) and controls (8, 10, 12 cells).
       Does a single 384-step plan break when the route alone is longer than ~15 cells?
  e1r: replication of e1 (added 11/26 before running it): S-small, S-large, L-small, 6 new scenarios each, picked
       with another seed among (A, G) pairs e1 did not use; same scoring and rules.
  e1 : one full obstacle b on a shortest A -> G route. 2x2 of the route left after blocking b
       (total <= 13 / >= 17 cells) and how much longer it is than the shortest route (extra 2-4 / >= 8);
       6 scenarios per group, picked with a fixed seed among distinct (A, G) pairs.

Each sample is scored like trigger_sampling.score:
  intact : in_wall <= 5 and max_jump < 0.5 (the plan itself is usable)
  clear  : no point touches the obstacle (always true in e0)
  valid  : intact and clear;  via_b : some point lies in the blocked cell (it took the shortest route)

    bash experiments/common/setup/run.sh python experiments/blocked/diag/budget_sampling.py --exp e1 --list
    bash experiments/common/setup/run.sh python experiments/blocked/diag/budget_sampling.py --exp e1 --idx 0 1 2
One JSON per scenario in results/diag/budget/<exp>/rows/, so several processes can share an output directory.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds  # noqa: E402

H = 384
GROUPS = (('S-small', lambda tot, ex: tot <= 13 and 2 <= ex <= 4), ('S-large', lambda tot, ex: tot <= 13 and ex >= 8),
          ('L-small', lambda tot, ex: tot >= 17 and 2 <= ex <= 4), ('L-large', lambda tot, ex: tot >= 17 and ex >= 8))
PER_GROUP = 6
PICK_SEED = 20261125
REP_SEED = 20261127
SEED_BASE = {'e0': 810000, 'e1': 810000, 'e1r': 830000}


def cells_between(maze, a, g, blocked=()):
    path = mu.bfs_path(maze, a, g, blocked)
    return None if path is None else len(path) - 1


def free_cells(maze):
    return [tuple(int(v) for v in c) for c in mu.free_cells(maze)]      # plain ints, for JSON


def e0_scenarios(maze, goal_cell):
    dist = {c: cells_between(maze, c, goal_cell) for c in free_cells(maze)}
    picks = [('far', c) for c in ((6, 2), (7, 2), (7, 1))] + [('mid', c) for c in ((1, 1), (5, 1), (5, 2))]
    picks += [('control', next(c for c in sorted(dist) if dist[c] == d)) for d in (8, 10, 12)]
    return [dict(id=f'e0_{a[0]}_{a[1]}', group=g, A=list(a), G=list(goal_cell), b=None, base=dist[a], total=dist[a], extra=0)
            for g, a in picks]


def e1_candidates(maze):
    """Every (A, G, b) of the plan, with the group it falls in (or None)."""
    free = free_cells(maze)
    dist = {a: {g: cells_between(maze, a, g) for g in free} for a in free}
    out = []
    for a in free:
        for g in free:
            if g == a:
                continue
            for b in free:
                if b in (a, g) or dist[a][b] < 2 or dist[a][b] + dist[b][g] != dist[a][g]:
                    continue
                tot = cells_between(maze, a, g, [b])
                if tot is None:
                    continue
                ex = tot - dist[a][g]
                grp = next((name for name, f in GROUPS if f(tot, ex)), None)
                out.append(dict(A=list(a), G=list(g), b=list(b), base=dist[a][g], total=tot, extra=ex, group=grp))
    return out


def e1_scenarios(maze, seed=PICK_SEED, groups=None, exclude=(), prefix='e1'):
    cands = e1_candidates(maze)
    rng = random.Random(seed)
    picked, counts = [], {}
    for name, _ in GROUPS:
        if groups is not None and name not in groups:
            continue
        mine = [c for c in cands if c['group'] == name]
        counts[name] = len(mine)
        pairs = sorted({(tuple(c['A']), tuple(c['G'])) for c in mine} - set(exclude))
        for a, g in rng.sample(pairs, PER_GROUP):
            options = sorted((c for c in mine if (tuple(c['A']), tuple(c['G'])) == (a, g)), key=lambda c: c['b'])
            c = rng.choice(options)
            picked.append(dict(id=f'{prefix}_{len(picked):02d}', **c))
    return picked, counts


def score(maze, plan, obstacle, b, margin):
    xy = plan[:, :2]
    in_wall = mu.count_points_in_wall(maze, xy)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    intact = bool(in_wall <= 5 and steps.max() < 0.5)
    clear = True if obstacle is None else ob.first_collision(xy, obstacle, margin) is None
    via_b = False if b is None else any(mu.xy_to_cell(p) == b for p in xy)
    return dict(intact=intact, clear=bool(clear), valid=bool(intact and clear), via_b=bool(via_b), in_wall=int(in_wall),
                max_jump=float(steps.max()), length=float(steps.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', choices=['e0', 'e1', 'e1r'], required=True)
    ap.add_argument('--list', action='store_true', help='only write scenarios.json')
    ap.add_argument('--idx', type=int, nargs='*', default=None, help='scenario indices to run (default all)')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--margin', type=float, default=0.05, help='check_margin of the probe configs')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--save', action='store_true', help='also save the plans (.npy)')
    ap.add_argument('--out', default='experiments/blocked/results/diag/budget')
    args = ap.parse_args()

    env = load_env('maze2d-large-v1')
    maze = env.maze_arr
    out = os.path.join(REPO, args.out, args.exp)
    os.makedirs(os.path.join(out, 'rows'), exist_ok=True)
    path = os.path.join(out, 'scenarios.json')
    if os.path.exists(path):                                       # frozen once written
        scen = json.load(open(path))['scenarios']
    else:
        if args.exp == 'e0':
            goal_cell = tuple(int(v) for v in mu.xy_to_cell(env._target))
            scen, meta = e0_scenarios(maze, goal_cell), dict(goal_cell=list(goal_cell), target=list(map(float, env._target)))
        elif args.exp == 'e1':
            scen, counts = e1_scenarios(maze)
            meta = dict(candidates=counts, per_group=PER_GROUP, pick_seed=PICK_SEED)
        else:
            used = {(tuple(s['A']), tuple(s['G'])) for s in json.load(open(os.path.join(REPO, args.out, 'e1', 'scenarios.json')))['scenarios']}
            scen, counts = e1_scenarios(maze, REP_SEED, ('S-small', 'S-large', 'L-small'), used, 'e1r')
            meta = dict(candidates=counts, per_group=PER_GROUP, pick_seed=REP_SEED, excluded_pairs=len(used))
        with open(path, 'w') as f:
            json.dump(dict(meta=meta, scenarios=scen), f, indent=1)
    for i, s in enumerate(scen):
        print(f"[budget {args.exp}] {i:2d} {s['id']} {s['group']:8s} A {s['A']} G {s['G']} b {s['b']} | "
              f"base {s['base']} total {s['total']} extra {s['extra']}", flush=True)
    if args.list:
        return

    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu', threads=args.threads)
    for i, s in enumerate(scen):
        dst = os.path.join(out, 'rows', f"{s['id']}.json")
        if (args.idx is not None and i not in args.idx) or os.path.exists(dst):
            continue
        state = np.r_[mu.cell_center(tuple(s['A'])), 0.0, 0.0]
        target = mu.cell_center(tuple(s['G']))
        b = None if s['b'] is None else tuple(s['b'])
        obstacle = None if b is None else ob.Obstacle.full(b)
        set_all_seeds(SEED_BASE[args.exp] + 1000 * i + H)
        samples, keep, dt = [], [], 0.0
        for _ in range(0, args.n, args.batch):
            plans, _, d = planner.plan(state, target, n=args.batch, horizon=H)
            dt += d
            for p in plans:
                samples.append(score(maze, p, obstacle, b, args.margin))
                keep.append(p.astype(np.float32))
        if args.save:
            np.save(os.path.join(out, 'rows', f"{s['id']}.npy"), np.stack(keep))
        k = {key: sum(x[key] for x in samples) for key in ('intact', 'clear', 'valid', 'via_b')}
        row = dict(**s, index=i, horizon=H, n=len(samples), plan_time_s=dt, **{f'n_{key}': v for key, v in k.items()},
                   median_length=float(np.median([x['length'] for x in samples])),
                   median_in_wall=float(np.median([x['in_wall'] for x in samples])), samples=samples)
        with open(dst, 'w') as f:
            json.dump(row, f)
        print(f"[budget {args.exp}] {s['id']} {s['group']:8s} total {s['total']:2d} extra {s['extra']:2d} | "
              f"intact {k['intact']:2d} clear {k['clear']:2d} valid {k['valid']:2d} via_b {k['via_b']:2d} /{len(samples)} | "
              f"length {row['median_length']:.1f} | {dt:.0f}s", flush=True)


if __name__ == '__main__':
    main()
