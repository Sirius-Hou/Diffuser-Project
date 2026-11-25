"""
Aggregate a run directory (one sub-folder per method) into episodes.csv, summary.json and figures.

    python experiments/blocked/analyze_blocked.py experiments/blocked/results/main
"""
import glob
import json
import os
import sys

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def stats(x):
    x = np.array([v for v in x if v is not None], dtype=float)
    if not len(x):
        return None
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), std=float(x.std()),
                min=float(x.min()), max=float(x.max()))


COLUMNS = ['method', 'scenario_id', 'seed', 'appear_t', 'success', 'first_success_step', 'obstacle_collision',
           'collision_step', 'failure_reason', 'final_dist_to_goal', 'steps', 'planning_time_s', 'n_plan_calls',
           'n_plans', 'n_replans', 'n_checks', 'n_triggers', 'n_all_collide', 'n_astar', 'n_astar_fail',
           'n_waypoints', 'n_segments', 'first_replan_after_obstacle', 'path_length', 'wall_contact_events',
           'wall_contact_steps', 'episode_time_s']


def main(run_dir):
    methods = [m for m in ['B0', 'B1', 'B2', 'Ours', 'Oracle'] if os.path.isdir(os.path.join(run_dir, m))]
    methods += sorted(m for m in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, m)) and m not in methods)
    recs = {}
    for m in methods:
        files = sorted(glob.glob(os.path.join(run_dir, m, 'episodes', '*.json')))
        recs[m] = [json.load(open(f)) for f in files]
    recs = {m: r for m, r in recs.items() if r}
    methods = list(recs)

    import csv
    with open(os.path.join(run_dir, 'episodes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for m in methods:
            for r in recs[m]:
                w.writerow([r.get(c) for c in COLUMNS])

    # only scenarios every method has finished, so the comparison is paired
    common = set.intersection(*[{r['scenario_id'] for r in recs[m]} for m in methods]) if methods else set()
    summary = dict(run_dir=run_dir, methods=methods, n_common_scenarios=len(common), by_method={})
    for m in methods:
        rs = [r for r in recs[m] if r['scenario_id'] in common]
        k = sum(r['success'] for r in rs)
        reasons = {}
        for r in rs:
            if not r['success']:
                reasons[r['failure_reason']] = reasons.get(r['failure_reason'], 0) + 1
        summary['by_method'][m] = dict(
            n=len(rs), n_success=k, success_rate=k / len(rs) if rs else None, success_95ci=wilson(k, len(rs)),
            obstacle_collision_rate=float(np.mean([r['obstacle_collision'] for r in rs])) if rs else None,
            failure_reasons=reasons,
            steps_to_success=stats([r['first_success_step'] for r in rs if r['success']]),
            planning_time_s=stats([r['planning_time_s'] for r in rs]),
            n_plans=stats([r['n_plans'] for r in rs]),
            n_replans=stats([r['n_replans'] for r in rs]),
            n_triggers=stats([r.get('n_triggers') for r in rs]),
            n_all_collide=stats([r.get('n_all_collide') for r in rs]),
            n_astar=stats([r.get('n_astar') for r in rs]),
            first_replan_after_obstacle=stats([r.get('first_replan_after_obstacle') for r in rs]),
            path_length_success=stats([r['path_length'] for r in rs if r['success']]),
            wall_contact_events=stats([r['wall_contact_events'] for r in rs]),
        )
    # paired table: which scenarios each method solves
    table = {sid: {m: next(r['success'] for r in recs[m] if r['scenario_id'] == sid) for m in methods}
             for sid in sorted(common)}
    summary['paired_success'] = table
    if 'B2' in methods and 'Ours' in methods:
        summary['ours_vs_b2'] = dict(
            both=sum(t['B2'] and t['Ours'] for t in table.values()),
            only_ours=sum(t['Ours'] and not t['B2'] for t in table.values()),
            only_b2=sum(t['B2'] and not t['Ours'] for t in table.values()),
            neither=sum(not t['B2'] and not t['Ours'] for t in table.values()))
    json.dump(summary, open(os.path.join(run_dir, 'summary.json'), 'w'), indent=1)

    print(f"{'method':6s} {'n':>3s} {'success':>8s} {'95% CI':>13s} {'collide':>8s} {'plans':>6s} {'plan s':>7s}  failures")
    for m in methods:
        b = summary['by_method'][m]
        ci = b['success_95ci']
        print(f"{m:6s} {b['n']:3d} {b['success_rate']:8.0%} {ci[0]:6.0%}-{ci[1]:5.0%} {b['obstacle_collision_rate']:8.0%} "
              f"{b['n_plans']['mean']:6.1f} {b['planning_time_s']['mean']:7.1f}  {b['failure_reasons']}")
    if 'ours_vs_b2' in summary:
        print('Ours vs B2 (paired):', summary['ours_vs_b2'])
    plot(run_dir, summary, methods)


def plot(run_dir, summary, methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bm = summary['by_method']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    x = np.arange(len(methods))
    sr = [bm[m]['success_rate'] for m in methods]
    lo = [sr[i] - bm[m]['success_95ci'][0] for i, m in enumerate(methods)]
    hi = [bm[m]['success_95ci'][1] - sr[i] for i, m in enumerate(methods)]
    axes[0].bar(x, sr, yerr=[lo, hi], capsize=4, color='tab:green')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('success rate (95% Wilson CI)')
    axes[1].bar(x, [bm[m]['obstacle_collision_rate'] for m in methods], color='tab:red')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('obstacle collision rate')
    axes[2].bar(x, [bm[m]['planning_time_s']['mean'] for m in methods], color='tab:blue')
    axes[2].set_title('planning time per episode (s, mean)')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
    fig.suptitle(f"{summary['n_common_scenarios']} paired scenarios", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'overview.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
