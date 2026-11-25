"""
R2 test (proposal risk R2, 開工說明 11/5-11/8): can B2 already get around the blockage just by
sampling more plans, without the maze map?

For each scenario: replay B0 up to B2's first trigger check after the obstacle appears
(t* = first multiple of k >= appear_t), then sample N = 16 plans from that exact state and count
  clear      no point within ball radius + margin of the obstacle
  wall_free  no plan point inside a wall cell (strict: corner-cutting also counts)
  valid      clear AND no point deeper than max_wall_depth inside a wall (= what best-of-N accepts;
             the plan switched corridor without tunnelling through a wall)
for each horizon setting. If almost every scenario has a valid plan among 16, B2 alone handles
full blockages and the main claim should move to compute / long detours (proposal fallback).

Also times plan(n) for n in probe.time_n (risk R5).

    python experiments/blocked/probe_b2.py --config experiments/blocked/configs/probe_b2.json
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
from experiments.common.methods import B0, B2  # noqa: E402
from experiments.common.rollout import run_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--scenarios')
    ap.add_argument('--out')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    pc = cfg['probe']
    env, planner, maze, contact = build(cfg, args.fake)
    data = json.load(open(os.path.join(REPO, args.scenarios or cfg['scenarios'])))
    scen = [s for s in data['scenarios'] if s['obstacles'][0]['kind'] in cfg.get('kinds', ['full'])]
    scen = scen[:args.limit or pc.get('limit') or len(scen)]
    out = os.path.join(REPO, args.out or cfg['out'])
    os.makedirs(os.path.join(out, 'scenarios'), exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')

    horizons = pc.get('horizons', ['full', 'euclid'])
    rows, timing = [], {}
    for si, s in enumerate(scen):
        appear = s['obstacles'][0]['appear_t']
        t_star = int(np.ceil(appear / pc['k']) * pc['k'])
        _, traj, ctx = run_episode(env, planner, B0(), s, cfg, maze, contact, stop_at_t=t_star)
        if ctx.t != t_star:          # collided / finished before the first check
            rows.append(dict(id=s['id'], skipped=f'episode ended at t={ctx.t}'))
            continue
        row = dict(id=s['id'], seed=s['seed'], t_check=t_star, ball_cell=list(mu.xy_to_cell(ctx.obs[:2])),
                   b0_remaining_collides=bool(ctx.remaining_collides()))
        all_plans = {}
        for hm in horizons:
            params = dict(cfg.get('method_params', {}).get('common', {}), replan_horizon=hm)
            H = B2(**params).replan_horizon(ctx)
            plans = ctx.sample(ctx.goal_xy, n=pc['n'], horizon=H, purpose=f'probe_{hm}')
            sc = ctx.score_plans(plans)
            clear = [x['collision'] is None for x in sc]
            wall_free = [x['wall_pts'] == 0 for x in sc]
            valid = [x['ok'] for x in sc]          # clear and no wall tunnelling (= what best-of-N accepts)
            row[hm] = dict(H=int(H or planner.horizon), n=len(sc), clear=int(sum(clear)), wall_free=int(sum(wall_free)),
                           valid=int(sum(valid)), max_wall_depth=[round(x['wall_depth'], 3) for x in sc],
                           seconds=ctx.plan_calls[-1]['seconds'])
            all_plans[hm] = (plans, valid, clear)
        if si == 0:
            for n in pc.get('time_n', []):
                ts = []
                for _ in range(2):
                    ctx.sample(ctx.goal_xy, n=n, purpose='timing')
                    ts.append(ctx.plan_calls[-1]['seconds'])
                timing[str(n)] = float(np.mean(ts))
        rows.append(row)
        render(os.path.join(out, 'scenarios', s['id'] + '.png'), maze, ctx, traj, all_plans, s, row)
        print(f"[probe] {s['id']:14s} t*={t_star:3d} " + ' | '.join(
            f"{hm}(H={row[hm]['H']}): valid {row[hm]['valid']}/{row[hm]['n']} clear {row[hm]['clear']} "
            f"{row[hm]['seconds']:.1f}s" for hm in horizons), flush=True)

    done = [r for r in rows if 'skipped' not in r]
    summary = dict(config=cfg, git_commit=git_commit(), fake=bool(args.fake), n_scenarios=len(done),
                   skipped=[r for r in rows if 'skipped' in r], plan_seconds_by_n=timing, by_horizon={})
    for hm in horizons:
        v = np.array([r[hm]['valid'] for r in done])
        c = np.array([r[hm]['clear'] for r in done])
        n = pc['n']
        summary['by_horizon'][hm] = dict(
            scenarios_with_any_valid=int((v > 0).sum()),
            frac_scenarios_with_any_valid=float((v > 0).mean()) if len(v) else None,
            mean_frac_valid_plans=float((v / n).mean()) if len(v) else None,
            mean_frac_clear_plans=float((c / n).mean()) if len(c) else None,
            # what best-of-N with a smaller N would see (expected, from the 16 samples)
            p_any_valid_with_N={str(k): float(np.mean([1 - _comb_none(int(x), n, k) for x in v])) if len(v) else None
                                for k in (1, 4, 8, 16)})
    summary['rows'] = rows
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: summary[k] for k in ('n_scenarios', 'plan_seconds_by_n', 'by_horizon')}, indent=1))


def _comb_none(valid, n, k):
    """P(no valid plan among k drawn without replacement from n samples with `valid` good ones)."""
    from math import comb
    return comb(n - valid, k) / comb(n, k) if k <= n else 0.0


def render(path, maze, ctx, traj, all_plans, s, row):
    import matplotlib.pyplot as plt
    hs = list(all_plans)
    fig, axes = plt.subplots(1, len(hs), figsize=(5.2 * len(hs), 4.2))
    for ax, hm in zip(np.atleast_1d(axes), hs):
        plans, valid, clear = all_plans[hm]
        mu.draw_maze(ax, maze)
        obsmod.draw_obstacles(ax, ctx.obstacles)
        for p, v, c in zip(plans, valid, clear):
            ax.plot(p[:, 1], p[:, 0], lw=0.8, alpha=0.8,
                    color='tab:green' if v else ('tab:olive' if c else 'tab:red'))
        r = traj['rollout_xy']
        ax.plot(r[:, 1], r[:, 0], color='tab:orange', lw=1.8)
        ax.set_title(f"{s['id']} t*={row['t_check']} H={row[hm]['H']} ({hm})\n"
                     f"green=valid {row[hm]['valid']}/{row[hm]['n']}, olive=clear but tunnels a wall, red=hits obstacle",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    main()
