"""
R3 / R5 measurements: how short can the frozen Diffuser's planning horizon be, and what does a batch cost?

Part 1 (R3): for start/goal cell pairs d cells apart, sample plans with horizon H = ratio * d (steps per
             cell) and execute them. Tells us which ratio a short leg needs (-> planner.horizon_for):
             too few steps per cell and the plan jumps, too many and it takes a longer route.
Part 2 (R5): wall-clock time of plan(n, H) for the batch sizes used by best-of-N.

    bash experiments/common/setup/run.sh python experiments/r3_horizon/run_r3.py --config experiments/r3_horizon/configs/smoke.json
    bash experiments/common/setup/run.sh python experiments/r3_horizon/run_r3.py --config experiments/r3_horizon/configs/main.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds, waypoint_action  # noqa: E402


def pick_pairs(maze, distances, pairs_per_distance, rng):
    """Random (start, goal, d) cell pairs whose shortest path is exactly d cells."""
    cells = mu.free_cells(maze)
    by_d = {d: [] for d in distances}
    for s in cells:
        for g in cells:
            p = mu.bfs_path(maze, s, g)
            if p is not None and len(p) - 1 in by_d:
                by_d[len(p) - 1].append((s, g))
    pairs = []
    for d in distances:
        for i in rng.choice(len(by_d[d]), size=min(pairs_per_distance, len(by_d[d])), replace=False):
            pairs.append((by_d[d][i][0], by_d[d][i][1], d))
    return pairs


def rollout(env, sequence, start_xy, goal_xy, n_steps):
    env.set_state(np.asarray(start_xy, dtype=np.float64), np.zeros(2))
    xys, contact = [np.asarray(start_xy, dtype=np.float64)], []
    for t in range(n_steps):
        obs, _, _, _ = env.step(waypoint_action(sequence, t, env.state_vector().copy()))
        xys.append(obs[:2].copy())
        contact.append(mu.wall_contact(env))
    return np.array(xys), np.array(contact, dtype=bool)


def first_within(dists, radius):
    idx = np.flatnonzero(dists <= radius)
    return int(idx[0]) if len(idx) else None


def pair_horizons(d, cfg, planner):
    """[(horizon, nominal steps per cell or None)] for a leg of d cells: one per ratio, plus the extras."""
    m = planner.horizon_multiple
    hs = {}
    for r in sorted(cfg['ratios'], reverse=True):       # if two ratios round to one horizon, keep the smaller
        hs[int(min(planner.horizon, max(cfg['min_horizon'], m * int(np.ceil(r * d / m)))))] = r
    for h in cfg.get('extra_horizons', []):
        hs.setdefault(h, None)
    return sorted(hs.items())


def extra_cells(xys, d):
    """Distinct cells a trajectory visits beyond the d + 1 of a shortest route (0 = some shortest route)."""
    return max(0, len(set(mu.cell_sequence(xys))) - (d + 1))


def run_horizons(cfg, env, planner, out):
    maze = env.maze_arr
    pairs = pick_pairs(maze, cfg['distances'], cfg['pairs_per_distance'], np.random.RandomState(cfg['seed']))
    r_goal, r_wp = cfg['success_radius'], cfg['waypoint_radius']
    rows = []
    for pi, (s, g, d) in enumerate(pairs):
        start_xy, goal_xy = mu.cell_center(s), mu.cell_center(g)
        obs = np.concatenate([start_xy, np.zeros(2)])
        for H, ratio in pair_horizons(d, cfg, planner):
            set_all_seeds(cfg['seed'] + 1000 * pi + H)
            plans, _, dt = planner.plan(obs, goal_xy, n=cfg['n_samples'], horizon=H)
            for k, seq in enumerate(plans):
                plan_xy = seq[:, :2]
                xys, contact = rollout(env, seq, start_xy, goal_xy, H + cfg['slack_steps'])
                dists = np.linalg.norm(xys - goal_xy, axis=1)
                rows.append(dict(
                    pair=pi, start_cell=f'{s[0]},{s[1]}', goal_cell=f'{g[0]},{g[1]}', cells=d, horizon=H, sample=k,
                    ratio=ratio, steps_per_cell=H / d,
                    plan_extra_cells=extra_cells(plan_xy, d), exec_extra_cells=extra_cells(xys, d),
                    plan_first_step_jump=float(np.linalg.norm(plan_xy[1] - plan_xy[0])),
                    plan_last_step_jump=float(np.linalg.norm(plan_xy[-1] - plan_xy[-2])),
                    plan_first_reach_idx=first_within(np.linalg.norm(plan_xy - goal_xy, axis=1), r_goal),
                    plan_wall_points=mu.count_points_in_wall(maze, plan_xy),
                    plan_max_step_jump=float(np.linalg.norm(np.diff(plan_xy, axis=0), axis=1).max()),
                    plan_path_length=mu.path_length(plan_xy),
                    reach_step=first_within(dists, r_goal), reach_step_wp=first_within(dists, r_wp),
                    reached=bool(dists.min() <= r_goal), reached_wp=bool(dists.min() <= r_wp),
                    dist_at_plan_end=float(dists[min(H - 1, len(dists) - 1)]), final_dist=float(dists[-1]),
                    wall_contact_events=mu.contact_events(contact),
                    planning_time_s=float(dt)))
            last = rows[-cfg['n_samples']:]
            print(f"[r3] pair {pi:2d} {s}->{g} d={d:2d} H={H:3d} | reached_wp {np.mean([r['reached_wp'] for r in last]):.2f} "
                  f"| on route {np.mean([r['exec_extra_cells'] == 0 for r in last]):.2f} | plan {dt:.2f}s", flush=True)
    df = pd.DataFrame(rows)
    df['good'] = df.reached_wp & (df.exec_extra_cells == 0)          # got there, along a shortest route
    for col in ('plan_first_reach_idx', 'reach_step', 'reach_step_wp'):      # None -> NaN even if all missing
        df[col] = pd.to_numeric(df[col])
    df.to_csv(os.path.join(out, 'horizon_runs.csv'), index=False)

    agg = dict(n=('good', 'size'), good=('good', 'mean'), reached=('reached', 'mean'),
               reached_wp=('reached_wp', 'mean'), on_route=('exec_extra_cells', lambda s: float((s == 0).mean())),
               plan_on_route=('plan_extra_cells', lambda s: float((s == 0).mean())),
               reach_step_wp_median=('reach_step_wp', 'median'),
               plan_in_wall=('plan_wall_points', lambda s: float((s > 0).mean())),
               first_jump=('plan_first_step_jump', 'median'), last_jump=('plan_last_step_jump', 'median'),
               contacts=('wall_contact_events', 'mean'))
    table = df.groupby(['cells', 'horizon']).agg(ratio=('ratio', 'first'), **agg).reset_index()
    table.to_csv(os.path.join(out, 'horizon_table.csv'), index=False)
    by_ratio = df[df.ratio.notna()].groupby('ratio').agg(**agg).reset_index()
    by_ratio.to_csv(os.path.join(out, 'ratio_table.csv'), index=False)

    # steps per cell with the highest "good" rate (ties: fewer steps); shortest horizon that works for a 1-cell leg
    best = by_ratio.sort_values(['good', 'ratio'], ascending=[False, True]).iloc[0]
    one = table[(table.cells == min(cfg['distances'])) & (table.good >= cfg['min_rate'])].horizon
    rec = dict(
        steps_per_cell=float(best.ratio), good_rate_at_best=float(best.good),
        min_horizon=int(one.min()) if len(one) else None,
        good_rate_by_ratio={str(r.ratio): float(r.good) for r in by_ratio.itertuples()},
        note='good = reached the waypoint radius within horizon + slack steps without visiting more cells than a shortest route. '
             'Copy steps_per_cell and min_horizon into experiments/blocked/configs.')
    return table, by_ratio, rec


def run_timing(cfg, planner):
    obs, goal = np.array([1.0, 1.0, 0.0, 0.0]), np.array([1.0, 3.0])
    rows = []
    planner.plan(obs, goal, n=1, horizon=cfg['timing']['horizons'][0])          # warm-up
    for H in cfg['timing']['horizons']:
        for n in cfg['timing']['batch_sizes']:
            ts = [planner.plan(obs, goal, n=n, horizon=H)[2] for _ in range(cfg['timing']['repeats'])]
            rows.append(dict(horizon=H, n=n, time_s_median=float(np.median(ts)), time_s_max=float(np.max(ts))))
            print(f"[r5] H={H:3d} n={n:2d} | {rows[-1]['time_s_median']:.2f}s per call", flush=True)
    return pd.DataFrame(rows)


def plot_table(table, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    piv = table[table.ratio.notna()].pivot(index='cells', columns='ratio', values='good')
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f'{piv.values[i, j]:.2f}', ha='center', va='center', color='w', fontsize=7)
    ax.set_xlabel('planning horizon / leg length (steps per cell)')
    ax.set_ylabel('leg length (cells)')
    ax.set_title('Fraction of rollouts that reach the waypoint along the shortest route')
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--device')
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.device:
        cfg['device'] = args.device
    out = os.path.join(REPO, cfg['out'])
    os.makedirs(out, exist_ok=True)

    planner = FrozenDiffuserPlanner(cfg['env'], cfg['diffusion_loadpath'], cfg['diffusion_epoch'],
                                    cfg['device'], threads=cfg.get('torch_threads', 0))
    env = load_env(cfg['env'])
    env.seed(cfg['seed'])
    env.reset()
    print(f'[r3] model epoch {planner.epoch} (H={planner.horizon}, multiple={planner.horizon_multiple}) '
          f'on {planner.device}', flush=True)

    t0 = time.perf_counter()
    table, by_ratio, rec = run_horizons(cfg, env, planner, out)
    timing = run_timing(cfg, planner)
    timing.to_csv(os.path.join(out, 'timing.csv'), index=False)
    plot_table(table, os.path.join(out, 'horizon_reach.png'))
    summary = dict(config=cfg, recommendation=rec, timing=timing.to_dict('records'),
                   total_time_s=time.perf_counter() - t0, **planner.meta())
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(table.round(2).to_string(index=False))
    print(by_ratio.round(2).to_string(index=False))
    print(timing.to_string(index=False))
    print('[r3] recommendation:', json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
