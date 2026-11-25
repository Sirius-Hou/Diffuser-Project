"""
Build the frozen scenario list (proposal: 篩選過「有繞路且走得完」的情境).

For each seed: replay B0 without obstacle (same seed -> same start and same first plan as every
method), pick when the obstacle appears and which corridor cell it blocks, keep the scenario only if

  * the ball's actual B0 route enters the cell between `lead_min` and `lead_max` steps after it
    appears (so B0 really collides, and there is time to react);
  * the cell is a corridor cell (exactly two opposite free neighbours), not the goal;
  * blocking it leaves another route from where the ball is to the goal (A*), and that route is a
    real detour (>= `min_detour_extra` cells longer than the unblocked shortest path);
  * the detour is plausibly finishable in time:
        appear_t + check period + steps_per_cell * detour cells + margin <= max_steps.

Run ONCE (needs the model), commit the JSON, and every method/teammate uses the same file.

    python experiments/blocked/make_scenarios.py --config experiments/blocked/configs/scenarios.json
    python experiments/blocked/make_scenarios.py --config experiments/blocked/configs/scenarios.json --fake   # logic test
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.blocked.runtime import REPO, build, git_commit  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as obsmod  # noqa: E402
from experiments.common import waypoints as wpmod  # noqa: E402
from experiments.common.methods import B0  # noqa: E402
from experiments.common.rollout import run_episode  # noqa: E402


def entry_times(rollout_xy):
    """First step at which the ball enters each cell (in order)."""
    out = []
    seen = set()
    for t, xy in enumerate(rollout_xy):
        c = mu.xy_to_cell(xy)
        if c not in seen:
            seen.add(c)
            out.append((t, c))
    return out


def pick(seed, rec, traj, maze, goal_cell, sc):
    rng = np.random.RandomState(10_000 + seed)
    r = traj['rollout_xy']
    if not rec['success']:
        return None, 'b0_fails_without_obstacle'
    t_goal = rec['first_success_step']
    lo, hi = sc['appear_t_range']
    hi = min(hi, t_goal - sc['lead_min'])
    if hi < lo:
        return None, 'reaches_goal_too_early'
    T = int(rng.randint(lo, hi + 1))
    cT = mu.xy_to_cell(r[T])
    unblocked = wpmod.bfs_dist(maze, cT).get(goal_cell)
    cands = []
    for t_c, c in entry_times(r[T:]):
        t_c += T
        if not (sc['lead_min'] <= t_c - T <= sc['lead_max']):
            continue
        if c == goal_cell or c == cT or obsmod.corridor_axis(maze, c) is None:
            continue
        ob = obsmod.Obstacle(cell=c, appear_t=T)
        if obsmod.disc_hits_rect(r[T], ob.rect(), sc['min_clearance_at_appear']):
            continue
        route = wpmod.astar_path(maze, cT, goal_cell, {c})
        if route is None:
            continue
        extra = (len(route) - 1) - unblocked
        if extra < sc['min_detour_extra']:
            continue
        need = T + sc['check_period'] + sc['steps_per_cell'] * (len(route) - 1) + sc['time_margin']
        if need > sc['max_steps']:
            continue
        wps = wpmod.select_waypoints(maze, route, {c})
        cands.append(dict(cell=c, b0_enters_at=t_c, detour_cells=len(route) - 1, detour_extra=extra,
                          n_waypoints=len(wps) - 1, waypoints=[list(w) for w in wps], est_steps_needed=need))
    if not cands:
        return None, 'no_valid_cell'
    ch = cands[rng.randint(len(cands))]
    return dict(appear_t=T, ball_cell_at_appear=list(cT), unblocked_cells_from_appear=unblocked, **ch), ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    sc = cfg['scenario']
    env, planner, maze, contact = build(cfg, args.fake)
    goal_cell = mu.xy_to_cell(env._target)
    out = os.path.join(REPO, args.out or cfg['out'])
    os.makedirs(os.path.dirname(out), exist_ok=True)

    scen, skipped = [], {}
    t0 = time.perf_counter()
    for seed in range(sc['seed_start'], sc['seed_start'] + sc['max_seeds']):
        if len([s for s in scen if s['obstacles'][0]['kind'] == 'full']) >= sc['n_scenarios']:
            break
        rec, traj, _ = run_episode(env, planner, B0(), {'seed': seed, 'obstacles': []}, cfg, maze, contact)
        info, why = pick(seed, rec, traj, maze, goal_cell, sc)
        if info is None:
            skipped[why] = skipped.get(why, 0) + 1
            print(f'[scen] seed {seed:3d}: skip ({why})', flush=True)
            continue
        c = tuple(info['cell'])
        base = dict(seed=seed, start_cell=rec['start_cell'], b0_success_without_obstacle=rec['success'],
                    b0_steps_without_obstacle=rec['first_success_step'],
                    **{k: v for k, v in info.items() if k != 'cell'})
        ob = obsmod.Obstacle(cell=c, appear_t=info['appear_t'])
        scen.append(dict(id=f'full_s{seed:03d}', obstacles=[ob.to_dict()], **base))
        for frac in sc.get('partial_fracs', []):
            axis = obsmod.corridor_axis(maze, c)
            side = 'lo' if np.random.RandomState(20_000 + seed).rand() < 0.5 else 'hi'
            pob = obsmod.Obstacle(cell=c, appear_t=info['appear_t'], kind='partial', frac=frac, side=side, axis=axis)
            scen.append(dict(id=f'partial{int(frac * 100)}_s{seed:03d}', obstacles=[pob.to_dict()], **base))
        print(f"[scen] seed {seed:3d}: obstacle {c} at t={info['appear_t']} (B0 enters at {info['b0_enters_at']}), "
              f"detour {info['detour_cells']} cells (+{info['detour_extra']}), {info['n_waypoints']} waypoint(s)", flush=True)

    meta = dict(config=cfg, fake=bool(args.fake or cfg.get('fake')), git_commit=git_commit(),
                created=time.strftime('%Y-%m-%d %H:%M:%S'), n_scenarios=len(scen), skipped=skipped,
                seconds=time.perf_counter() - t0)
    json.dump(dict(meta=meta, scenarios=scen), open(out, 'w'), indent=1)
    print(f'[scen] {len(scen)} scenarios -> {os.path.relpath(out, REPO)} | skipped {skipped}')
    render_overview(out[:-5] + '.png', maze, [s for s in scen if s['id'].startswith('full')], env._target)


def render_overview(path, maze, scen, goal_xy, max_n=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scen = scen[:max_n]
    if not scen:
        return
    cols = 6
    rows = int(np.ceil(len(scen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.0))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')
    for ax, s in zip(np.atleast_1d(axes).ravel(), scen):
        mu.draw_maze(ax, maze)
        obsmod.draw_obstacles(ax, [obsmod.Obstacle.from_dict(o) for o in s['obstacles']], color='red')
        sc = [int(v) for v in s['start_cell'].split(',')]
        bc = s['ball_cell_at_appear']
        ax.scatter([sc[1]], [sc[0]], color='green', s=12)
        ax.scatter([bc[1] - 0.2], [bc[0] - 0.2], color='tab:orange', s=12)
        for w in s['waypoints'][:-1]:
            ax.scatter([w[1] - 0.2], [w[0] - 0.2], marker='*', color='gold', edgecolor='k', s=40, linewidths=0.4)
        ax.scatter([goal_xy[1]], [goal_xy[0]], color='magenta', s=12)
        ax.set_title(f"{s['id']} t={s['appear_t']} +{s['detour_extra']}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('scenarios: green=start, orange=ball when obstacle appears, red=obstacle, star=A* waypoint', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
