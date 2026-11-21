"""
Aggregate the blocked-corridor runs of every method into one comparison.

    bash experiments/common/setup/run.sh python experiments/blocked/analyze_blocked.py experiments/blocked/results/main

Reads <run>/<method>/episodes/seedNNN.json; writes episodes.csv, summary.json, summary.md and comparison.png
into <run>. Scenarios that were excluded in any method are dropped from every method, so the rates are paired.

Rates are also split by case: a full blockage either forces a longer route ("full, longer detour": the
blocked cell is on every shortest route) or leaves a route of the same length ("full, same length"); a
partial blockage either has a route around the cell ("partial, detour exists") or can only be passed
through its gap ("partial, no detour").
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from experiments.b0_baseline.analyze_b0 import wilson  # noqa: E402

ORDER = ['B0', 'B1', 'B2', 'ours', 'astar_line', 'oracle']


def method_row(df):
    n, k = len(df), int(df.success.sum())
    lo, hi = wilson(k, n)
    succ = df[df.success]
    row = dict(n=n, n_success=k, success_rate=k / n if n else None, ci_lo=lo, ci_hi=hi,
               obstacle_collision_rate=float(df.obstacle_collision.mean()) if n else None,
               steps_to_success_median=float(succ.first_success_step.median()) if k else None,
               path_length_success_mean=float(succ.path_length.mean()) if k else None,
               replans_mean=float(df.n_replans.mean()), samples_mean=float(df.n_samples.mean()),
               planning_time_s_mean=float(df.planning_time_s.mean()),
               wall_contact_events_mean=float(df.wall_contact_events.mean()),
               failure_reasons=df[~df.success].failure_reason.value_counts().to_dict())
    return row


def main(rdir):
    rows = []
    for m in ORDER:
        for f in sorted(glob.glob(os.path.join(rdir, m, 'episodes', 'seed[0-9][0-9][0-9].json'))):
            r = json.load(open(f))
            r.pop('events', None)
            rows.append(r)
    assert rows, f'no episodes under {rdir}'
    df = pd.DataFrame(rows)
    longer = df.detour_cells.notna() & (df.detour_cells > df.base_cells)
    df['case'] = np.where(df.obstacle != 'full',
                          np.where(df.detour_cells.notna(), 'partial, detour exists', 'partial, no detour'),
                          np.where(longer, 'full, longer detour', 'full, same length'))
    methods = [m for m in ORDER if m in set(df.method)]
    df.to_csv(os.path.join(rdir, 'episodes.csv'), index=False)

    bad = set(df[df.excluded_reason != ''].seed)
    common = set.intersection(*[set(df[df.method == m].seed) for m in methods]) - bad
    paired = df[df.seed.isin(common)]
    summary = dict(methods=methods, n_paired_scenarios=len(common), dropped_seeds=sorted(int(s) for s in bad),
                   overall={}, by_obstacle={})
    for m in methods:
        d = paired[paired.method == m]
        summary['overall'][m] = method_row(d)
        summary['by_obstacle'][m] = {kind: method_row(d[d.case == kind]) for kind in sorted(set(d.case))}

    # R2: when the trigger first fired, how many of the N samples were clear of the obstacle?
    probe = {}
    for m in ('B2', 'ours'):
        d = paired[(paired.method == m) & paired.first_replan_n_clear.notna()]
        for kind in sorted(set(d.case)):
            dk = d[d.case == kind]
            probe[f'{m}/{kind}'] = dict(n_triggered=len(dk),
                                        best_of_n_found_clear_plan=float((dk.first_replan_n_clear > 0).mean()),
                                        mean_clear_samples=float(dk.first_replan_n_clear.mean()))
    summary['best_of_n_at_first_trigger'] = probe
    with open(os.path.join(rdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    lines = [f'Paired scenarios: {len(common)} (dropped seeds: {sorted(int(s) for s in bad) or "none"})', '',
             '| Method | Obstacle | n | Success | 95% CI | Obstacle hit | Steps (median) | Replans | Samples | Plan time (s) | Wall contacts |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in methods:
        for kind, r in [('all', summary['overall'][m])] + list(summary['by_obstacle'][m].items()):
            if not r['n']:
                continue
            steps = f"{r['steps_to_success_median']:.0f}" if r['steps_to_success_median'] is not None else '-'
            lines.append(f"| {m} | {kind} | {r['n']} | {r['n_success']}/{r['n']} = {100 * r['success_rate']:.0f}% "
                         f"| {100 * r['ci_lo']:.0f}-{100 * r['ci_hi']:.0f}% | {100 * r['obstacle_collision_rate']:.0f}% "
                         f"| {steps} | {r['replans_mean']:.1f} | {r['samples_mean']:.1f} "
                         f"| {r['planning_time_s_mean']:.1f} | {r['wall_contact_events_mean']:.1f} |")
    with open(os.path.join(rdir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps(probe, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    kinds = ['all'] + sorted(set(paired.case))
    width = 0.8 / len(kinds)
    for j, kind in enumerate(kinds):
        rs = [summary['overall'][m] if kind == 'all' else summary['by_obstacle'][m].get(kind) for m in methods]
        rate = [100 * r['success_rate'] if r and r['n'] else 0 for r in rs]
        err = [[max(0.0, 100 * (r['success_rate'] - r['ci_lo'])) if r and r['n'] else 0 for r in rs],
               [max(0.0, 100 * (r['ci_hi'] - r['success_rate'])) if r and r['n'] else 0 for r in rs]]
        axes[0].bar([i + j * width for i in range(len(methods))], rate, width, yerr=err, capsize=2, label=kind)
    axes[0].set_xticks([i + 0.4 - width / 2 for i in range(len(methods))])
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('success rate (%)')
    axes[0].set_ylim(0, 105)
    axes[0].legend(title='obstacle', fontsize=7)
    axes[1].bar(methods, [summary['overall'][m]['samples_mean'] for m in methods], color='0.5')
    axes[1].set_ylabel('plans sampled per episode')
    fig.suptitle(f'Blocked corridor, {len(common)} paired scenarios (bars: Wilson 95% CI)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(rdir, 'comparison.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
