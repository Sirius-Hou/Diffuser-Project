"""
Aggregate per-episode B0 results into episodes.csv, summary.json, by_start_cell.csv and figures,
and evaluate the B0 acceptance criteria.

    bash experiments/common/env/run.sh python experiments/b0_baseline/analyze_b0.py experiments/b0_baseline/results/main
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402

# Acceptance criteria for B0 (see README section 4)
MIN_SUCCESS_RATE = 0.80
EXPECTED_EPISODES = 30


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def stats(s):
    s = pd.Series(s).dropna().astype(float)
    if len(s) == 0:
        return None
    return dict(n=int(len(s)), mean=float(s.mean()), std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                median=float(s.median()), min=float(s.min()), max=float(s.max()))


def main(rdir):
    files = sorted(glob.glob(os.path.join(rdir, 'episodes', 'seed[0-9][0-9][0-9].json')))
    df = pd.DataFrame([json.load(open(f)) for f in files]).sort_values('seed').reset_index(drop=True)
    meta = json.load(open(os.path.join(rdir, 'run_meta.json')))
    df.to_csv(os.path.join(rdir, 'episodes.csv'), index=False)

    n = len(df)
    k = int(df.success.sum())
    succ, fail = df[df.success], df[~df.success]
    df['steps_per_cell'] = df.first_success_step / df.shortest_path_cells.replace(0, np.nan)
    df['path_efficiency'] = df.path_length_to_success / df.plan_path_length
    late = df[df.dist_to_goal_at_plan_end > 0.5]       # still not at goal when the 384-step plan ran out

    summary = dict(
        method='B0 (Diffuser, plan once at t=0, no replanning)',
        setting='maze2d-large-v1, fixed goal (7,9), random start (env.reset), NO obstacle',
        n_episodes=n, seeds=df.seed.tolist(),
        model=dict(epoch=meta['diffusion_epoch'], horizon=meta['horizon'],
                   n_diffusion_steps=meta['n_diffusion_steps'], device=meta['device'],
                   torch_threads=meta['torch_threads']),
        success_rate=k / n, n_success=k, success_rate_95ci_wilson=wilson(k, n),
        success_at_end_rate=float(df.success_at_end.mean()),
        reached_then_left=int(df.reached_then_left.sum()),
        final_dist_to_goal=stats(df.final_dist_to_goal),
        final_dist_to_goal_success=stats(succ.final_dist_to_goal),
        final_dist_to_goal_failure=stats(fail.final_dist_to_goal),
        first_success_step=stats(succ.first_success_step),
        planning_time_s=stats(df.planning_time_s),
        episode_time_s=stats(df.episode_time_s),
        n_replans_total=int(df.n_replans.sum()),
        path_length=stats(df.path_length),
        path_length_to_success=stats(succ.path_length_to_success),
        plan_path_length=stats(df.plan_path_length),
        path_to_success_over_plan_length=stats(df.path_efficiency),
        shortest_path_cells=stats(df.shortest_path_cells),
        steps_per_cell_success=stats(df.loc[df.success, 'steps_per_cell']),
        plan_detour_cells=stats(df.plan_detour_cells),
        wall_contact_events=stats(df.wall_contact_events),
        wall_contact_steps=stats(df.wall_contact_steps),
        episodes_with_wall_contact=int((df.wall_contact_events > 0).sum()),
        wall_contact_events_success=stats(succ.wall_contact_events),
        wall_contact_events_failure=stats(fail.wall_contact_events),
        obstacle_collisions_total=int(df.obstacle_collisions.sum()),
        plans_with_center_in_wall=int((df.plan_wall_points > 0).sum()),
        plans_with_ball_touching_wall=int((df.plan_wall_touch_points > 0).sum()),
        plan_end_dist_to_goal=stats(df.plan_end_dist_to_goal),
        track_err_mean_first_H=stats(df.track_err_mean_first_H),
        dist_to_goal_at_plan_end=stats(df.dist_to_goal_at_plan_end),
        episodes_not_at_goal_when_plan_ended=int(len(late)),
        success_rate_when_not_at_goal_at_plan_end=(float(late.success.mean()) if len(late) else None),
        d4rl_normalized_score=stats(df.d4rl_normalized_score),
        failure_reasons=fail.failure_reason.value_counts().to_dict(),
        failures=fail[['seed', 'start_cell', 'final_dist_to_goal', 'failure_reason', 'wall_contact_events',
                       'dist_to_goal_at_plan_end', 'stall_start_step', 'plan_wall_points',
                       'track_err_mean_first_H']].to_dict(orient='records'),
    )
    # per start cell
    by_cell = df.groupby('start_cell').agg(n=('seed', 'size'), success=('success', 'mean'),
                                           shortest_cells=('shortest_path_cells', 'first'),
                                           mean_contacts=('wall_contact_events', 'mean'),
                                           mean_first_success_step=('first_success_step', 'mean')).reset_index()
    by_cell = by_cell.sort_values('shortest_cells')
    by_cell.to_csv(os.path.join(rdir, 'by_start_cell.csv'), index=False)
    summary['by_start_cell'] = by_cell.to_dict(orient='records')

    # correlations worth flagging
    summary['corr_shortest_cells_vs_first_success_step'] = float(
        succ[['shortest_path_cells', 'first_success_step']].corr().iloc[0, 1]) if len(succ) > 2 else None
    summary['corr_contacts_vs_track_err'] = float(df[['wall_contact_events', 'track_err_mean_first_H']].corr().iloc[0, 1])
    summary['corr_shortest_cells_vs_plan_first_reach_idx'] = float(
        df[['shortest_path_cells', 'plan_first_reach_idx']].corr().iloc[0, 1])
    summary['plan_first_reach_idx'] = stats(df.plan_first_reach_idx)
    summary['git_commit'] = meta.get('git_commit')

    # acceptance criteria
    key_cols = ['success', 'final_dist_to_goal', 'planning_time_s', 'path_length', 'wall_contact_events',
                'failure_reason']
    acc = {
        'all_episodes_completed': bool(n >= EXPECTED_EPISODES),
        'all_metrics_recorded': bool(df[key_cols[:-1]].notna().all().all()),
        f'success_rate_ge_{MIN_SUCCESS_RATE}': bool(k / n >= MIN_SUCCESS_RATE),
        'every_failure_has_reason': bool((fail.failure_reason.fillna('') != '').all()),
    }
    rp = os.path.join(os.path.dirname(os.path.abspath(rdir)), 'smoke', 'repro_check.json')
    acc['seeds_reproducible'] = bool(json.load(open(rp))['all_identical']) if os.path.exists(rp) else None
    acc['ALL_PASSED'] = all(v for v in acc.values() if v is not None)
    summary['acceptance'] = acc

    with open(os.path.join(rdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    # ---------------- figures
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    c = np.where(df.success, 'tab:green', 'tab:red')
    axs[0, 0].bar(df.seed, df.final_dist_to_goal, color=c); axs[0, 0].axhline(0.5, ls='--', c='k', lw=0.8)
    axs[0, 0].set_title('final distance to goal (red = fail)'); axs[0, 0].set_xlabel('seed')
    axs[0, 1].scatter(df.shortest_path_cells, df.first_success_step, c='tab:green')
    axs[0, 1].axhline(meta['horizon'], ls='--', c='gray', lw=0.8, label='plan horizon 384')
    for _, r in fail.iterrows():
        axs[0, 1].scatter(r.shortest_path_cells, 800, c='tab:red', marker='x')
    axs[0, 1].set_xlabel('BFS shortest path (cells)'); axs[0, 1].set_ylabel('steps to reach goal (x = never)')
    axs[0, 1].set_title('time to goal vs. distance'); axs[0, 1].legend(fontsize=7)
    axs[0, 2].hist(df.planning_time_s, bins=15, color='tab:blue'); axs[0, 2].set_title('planning time per plan (s, CPU)')
    axs[1, 0].bar(df.seed, df.wall_contact_events, color=c); axs[1, 0].set_title('wall contact events'); axs[1, 0].set_xlabel('seed')
    axs[1, 1].scatter(df.track_err_mean_first_H, df.wall_contact_events, c=c)
    axs[1, 1].set_xlabel('mean tracking error vs plan (first 384 steps)'); axs[1, 1].set_ylabel('wall contact events')
    axs[1, 1].set_title('tracking error vs contacts')
    axs[1, 2].scatter(df.shortest_path_cells, df.dist_to_goal_at_plan_end, c=c)
    axs[1, 2].axhline(0.5, ls='--', c='k', lw=0.8)
    axs[1, 2].set_xlabel('BFS shortest path (cells)'); axs[1, 2].set_ylabel('dist to goal at t=383')
    axs[1, 2].set_title('where the agent is when the plan runs out')
    fig.tight_layout(); fig.savefig(os.path.join(rdir, 'b0_overview.png'), dpi=110); plt.close(fig)

    # start-cell success map
    try:
        import d4rl.pointmaze.maze_model as mm
        maze = mm.parse_maze(mm.LARGE_MAZE)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        img = np.where(maze == mu.WALL, 0.35, 1.0)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        for _, r in by_cell.iterrows():
            w, h = map(int, r.start_cell.split(','))
            ax.add_patch(plt.Rectangle((h - .5, w - .5), 1, 1, color=plt.cm.RdYlGn(r.success), alpha=0.8))
            ax.text(h, w, f"{int(r.n)}", ha='center', va='center', fontsize=8)
        ax.text(9, 7, 'G', ha='center', va='center', fontsize=12, weight='bold', color='magenta')
        ax.set_title('B0 success rate by start cell (number = episodes)')
        ax.set_xticks(range(maze.shape[1])); ax.set_yticks(range(maze.shape[0]))
        fig.tight_layout(); fig.savefig(os.path.join(rdir, 'b0_start_cells.png'), dpi=110); plt.close(fig)
    except Exception as e:  # figure is optional
        print('start-cell map skipped:', e)

    # failure montage
    pngs = [os.path.join(rdir, 'episodes', f'seed{s:03d}.png') for s in fail.seed]
    pngs = [p for p in pngs if os.path.exists(p)]
    if pngs:
        import imageio
        ims = [imageio.imread(p)[..., :3] for p in pngs]
        imageio.imwrite(os.path.join(rdir, 'b0_failures.png'), np.concatenate(ims, axis=1))

    print(json.dumps({k: summary[k] for k in ('n_episodes', 'success_rate', 'success_rate_95ci_wilson',
                                              'failure_reasons', 'episodes_not_at_goal_when_plan_ended',
                                              'success_rate_when_not_at_goal_at_plan_end', 'acceptance')},
                     indent=1))
    print(by_cell.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'experiments', 'b0_baseline', 'results', 'main'))
