"""
Every corridor of the maze as one junction-mode leg, without obstacles: can the frozen Diffuser follow it,
and can a straight line?

The nodes are the free cells with other than two open neighbours (junctions and dead ends); a leg is the
corridor between two neighbouring nodes, in both directions. The ball starts at rest at the first node's
centre and must come within `radius` of the second's, as a waypoint leg of ours does (probe_junctions):

  diffuser     : n samples with horizon 32 x cells, conditioned on the position only (leg_start_at_rest),
                 chosen as ours chooses leg plans (fewest cells off the corridor, then fewest points in a
                 wall, then the smallest jump at the end)
  line         : one constant-speed straight line to the node (astar_line with line_turns false)
  line_corners : the line through the corridor's turns (astar_line)

A plan that has run out (leg_grace steps after its last point) is replaced from the ball's state, at most
max_plans plans per leg, all tracked with the original waypoint_action.

    bash experiments/common/setup/run.sh python experiments/blocked/diag/leg_test.py
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_graph as mg  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds, waypoint_action  # noqa: E402

METHODS = ('diffuser', 'line', 'line_corners')


def corridor_legs(maze):
    """(path of cells) for every corridor between two neighbouring nodes, in both directions."""
    nodes = {c for c in mu.free_cells(maze) if mg.degree(maze, c) != 2}
    legs = []
    for a in sorted(nodes):
        for d in mu.NEIGHBORS4:
            path = [a, (a[0] + d[0], a[1] + d[1])]
            if not mu.is_free(maze, path[-1]):
                continue
            while path[-1] not in nodes:
                nxt = [(path[-1][0] + e[0], path[-1][1] + e[1]) for e in mu.NEIGHBORS4]
                path.append(next(c for c in nxt if mu.is_free(maze, c) and c != path[-2]))
            legs.append(path)
    return legs


def choose(maze, plans, route):
    route = set(route)
    keys = [(len(set(mu.cell_sequence(p[:, :2])) - route), mu.count_points_in_wall(maze, p[:, :2]),
             float(np.linalg.norm(p[-1, :2] - p[-2, :2]))) for p in plans]
    return plans[int(min(range(len(plans)), key=keys.__getitem__))]


def make_plan(method, planner, maze, state, path, goal, h, n, dt):
    if method == 'diffuser':
        plans, _, _ = planner.plan(np.r_[state[:2], 0.0, 0.0], goal, n=n, horizon=h)
        return choose(maze, plans, path)
    cur = mu.xy_to_cell(state[:2])
    i = path.index(cur) if cur in path else 0
    rest = path[i:]
    turns = [mu.cell_center(rest[j]) for j in range(1, len(rest) - 1) if mg.is_corner(rest, j)]
    return mg.line_plan([state[:2], *(turns if method == 'line_corners' else []), goal], h, dt)


def run_leg(method, env, planner, path, args):
    maze = env.maze_arr
    cells = len(path) - 1
    h = planner.horizon_for(cells, args.steps_per_cell, args.min_horizon)
    goal = mu.cell_center(path[-1])
    env.reset()
    env.set_state(mu.cell_center(path[0]), np.zeros(2))
    state = env.state_vector().copy()
    seq, t0 = make_plan(method, planner, maze, state, path, goal, h, args.n, env.unwrapped.dt), 0
    plans, first = 1, seq
    contacts, reached = [], None
    for t in range(args.max_plans * (h + args.leg_grace)):
        if np.linalg.norm(state[:2] - goal) <= args.radius:
            reached = t
            break
        if t - t0 >= len(seq) - 1 + args.leg_grace:
            if plans == args.max_plans:
                break
            seq, t0, plans = make_plan(method, planner, maze, state, path, goal, h, args.n, env.unwrapped.dt), t, plans + 1
        env.step(waypoint_action(seq, t - t0, state))
        state = env.state_vector().copy()
        contacts.append(mu.wall_contact(env))
    return dict(method=method, start=list(path[0]), end=list(path[-1]), cells=cells, horizon=h,
                turns=sum(mg.is_corner(path, j) for j in range(1, cells)),
                success=reached is not None, steps=reached, plans=plans,
                first_plan_in_wall=mu.count_points_in_wall(maze, first[:, :2]),
                first_plan_off_route=len(set(mu.cell_sequence(first[:, :2])) - set(path)),
                wall_contact_steps=int(np.sum(contacts)), final_dist=float(np.linalg.norm(state[:2] - goal)),
                final_cell=list(mu.xy_to_cell(state[:2])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=8, help='n_samples_leg of the probe configs')
    ap.add_argument('--steps_per_cell', type=int, default=32)
    ap.add_argument('--min_horizon', type=int, default=32)
    ap.add_argument('--radius', type=float, default=0.5)
    ap.add_argument('--leg_grace', type=int, default=20)
    ap.add_argument('--max_plans', type=int, default=3)
    ap.add_argument('--out', default='experiments/blocked/results/diag/leg_test')
    args = ap.parse_args()
    out = os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu')
    env = load_env('maze2d-large-v1')
    legs = corridor_legs(env.maze_arr)
    print(f'[legs] {len(legs)} directed corridor legs', flush=True)
    rows = []
    for i, path in enumerate(legs):
        for m in METHODS:
            set_all_seeds(800000 + i)
            r = run_leg(m, env, planner, path, args)
            rows.append(r)
            print(f"[legs] {i:2d} {tuple(r['start'])}->{tuple(r['end'])} cells {r['cells']:2d} turns {r['turns']} "
                  f"| {m:12s} | {'ok ' if r['success'] else 'X  '} steps {r['steps']} plans {r['plans']} "
                  f"| final {tuple(r['final_cell'])} {r['final_dist']:.2f}", flush=True)
            with open(os.path.join(out, 'rows.json'), 'w') as f:
                json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
