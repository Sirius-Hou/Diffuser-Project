"""
Build the scenario list shared by every method: (seed, start state, t=0 plan, obstacle).

For each seed we reset the env and sample the t=0 plan exactly as B0 does, then put the obstacle
on a cell that this plan crosses after the appearance time. Every method later starts from the
stored state and the stored plan, so all methods face the same blockage on the route the agent
is really taking. Seeds with no usable cell are listed under "excluded" with the reason.

  full    : the whole cell; kept only if A* finds a detour that fits in the remaining steps
  partial : `partial_frac` of the corridor width, on the side the plan uses; straight corridor cells only

    bash experiments/common/setup/run.sh python experiments/blocked/make_scenarios.py --config experiments/blocked/configs/smoke.json
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_graph as mg  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds  # noqa: E402


def detour_steps(planner, maze, path, blocked, cfg):
    """Steps subgoal chaining needs for `path`: one plan per leg, each with its scaled horizon."""
    legs = mg.leg_lengths(mg.waypoints(maze, path, blocked, cfg['max_leg_cells']))
    return sum(planner.horizon_for(c, cfg['steps_per_cell'], cfg['min_horizon']) for c in legs)


def candidates(kind, maze, planner, plan_xy, t_appear, goal_cell, max_steps, cfg):
    """Obstacles of `kind` on cells the plan crosses at least min_ahead_cells after t_appear."""
    cells = mu.cell_sequence(plan_xy[t_appear:])
    cur, near = cells[0], set(cells[:cfg['min_ahead_cells']])
    out = []
    for c in dict.fromkeys(cells[cfg['min_ahead_cells']:]):          # unique, in plan order
        if c in near or c == goal_cell:
            continue
        base = mu.bfs_path(maze, cur, goal_cell)
        if kind == 'full':
            path = mg.astar(maze, cur, goal_cell, [c])
            if path is None:
                continue
            need = detour_steps(planner, maze, path, [c], cfg)
            if t_appear + need + cfg['budget_slack_steps'] > max_steps:
                continue
            obstacle = ob.Obstacle.full(c, t_appear)
            detour = len(path) - 1
        else:
            axis = ob.corridor_axis(maze, c)
            if axis is None:
                continue
            w = 1 - axis
            inside = np.array([p for p in plan_xy[t_appear:] if mu.xy_to_cell(p) == c])
            side = -1 if inside[:, w].mean() < mu.cell_center(c)[w] else 1
            obstacle = ob.Obstacle.partial(c, axis, side, cfg['partial_frac'], t_appear)
            path = mg.astar(maze, cur, goal_cell, [c])
            detour = len(path) - 1 if path else None
        if not ob.collides(plan_xy[t_appear:], obstacle):
            continue
        out.append(dict(obstacle=obstacle.to_dict(), cell_at_appear=list(cur),
                        base_cells=len(base) - 1 if base else None, detour_cells=detour))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--device')
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    sc_cfg = dict(cfg, **cfg['scenarios'])           # scenario options + the shared leg-horizon settings
    out = os.path.join(REPO, cfg['out'])
    os.makedirs(out, exist_ok=True)

    planner = FrozenDiffuserPlanner(cfg['env'], cfg['diffusion_loadpath'], cfg['diffusion_epoch'],
                                    args.device or cfg['device'], threads=cfg.get('torch_threads', 0))
    env = load_env(cfg['env'])
    maze = env.maze_arr
    want = {'full': sc_cfg['n_full'], 'partial': sc_cfg['n_partial']}
    scenarios, excluded, plans = [], [], {}
    seed = sc_cfg['seed_start']
    while sum(want.values()) > 0 and seed < sc_cfg['seed_start'] + sc_cfg['max_seeds']:
        set_all_seeds(seed)
        env.seed(seed)
        obs = env.reset()
        qpos, qvel = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
        target = np.array(env._target, dtype=np.float64)
        goal_cell = mu.xy_to_cell(target)
        plan, _, dt = planner.plan(obs, target, n=1)
        plan = plan[0]
        rng = np.random.RandomState(100003 + seed)                   # separate from the sampling RNG
        t_appear = int(rng.choice(sc_cfg['t_appear']))
        chosen = None
        for kind in ('full', 'partial'):                             # full scenarios are the rare ones
            if want[kind] == 0:
                continue
            cands = candidates(kind, maze, planner, plan[:, :2], t_appear, goal_cell,
                               env.max_episode_steps, sc_cfg)
            if cands:
                chosen = cands[rng.randint(len(cands))]
                want[kind] -= 1
                break
        start_cell = mu.xy_to_cell(obs[:2])
        if chosen is None:
            excluded.append(dict(seed=seed, start_cell=list(start_cell), t_appear=t_appear,
                                 reason='no_usable_cell_on_plan'))
            print(f'[scen] seed {seed:3d} | start {start_cell} | excluded', flush=True)
        else:
            scenarios.append(dict(seed=seed, qpos=qpos.tolist(), qvel=qvel.tolist(), start_cell=list(start_cell),
                                  goal_cell=list(goal_cell), t_appear=t_appear, plan0_time_s=float(dt), **chosen))
            plans[f'seed{seed:03d}'] = plan
            o = chosen['obstacle']
            print(f"[scen] seed {seed:3d} | start {start_cell} | {o['kind']:7s} at {tuple(o['cell'])} t={t_appear} "
                  f"| route {chosen['base_cells']} -> detour {chosen['detour_cells']} cells", flush=True)
        seed += 1

    with open(os.path.join(out, 'scenarios.json'), 'w') as f:
        json.dump(dict(config=cfg, device=planner.device, scenarios=scenarios, excluded=excluded,
                       unfilled=want), f, indent=2)
    np.savez_compressed(os.path.join(out, 'plans0.npz'), **plans)
    print(f'[scen] {len(scenarios)} scenarios, {len(excluded)} seeds excluded, unfilled {want} -> '
          f'{os.path.relpath(out, REPO)}', flush=True)


if __name__ == '__main__':
    main()
