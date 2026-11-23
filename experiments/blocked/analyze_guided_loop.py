"""
Closed-loop comparison of B2 and b2_guided (Guidance實驗計畫_2026-11-22.md, section 11; exploratory).

    python experiments/blocked/analyze_guided_loop.py partial_fresh probe_partial      (plain Python)

For each results/<name>: paired scenarios (seeds excluded in any method are dropped from all), success with a
Wilson CI, exact McNemar on the discordant pairs, split by whether another route exists, and where each
obstacle collision came from:
  execution : the last replan before the hit had a sample clear of the obstacle (the ball left a clear plan)
  planning  : no clear sample at the last replan (the plan that was followed went through the obstacle)
Writes results/<name>/guided_loop.{md,json}.
"""
import glob
import json
import os
import sys

from analyze_main import RES, mcnemar_exact, wilson

METHODS = ('B0', 'B2', 'b2_guided', 'oracle')


def layer(r):
    if not r['obstacle_collision']:
        return None
    replans = [e for e in r.get('events', []) if e['event'] == 'replan' and e['t'] <= r['obstacle_collision_step']]
    if not replans:
        return 'no replan'
    return 'execution' if replans[-1]['n_clear'] > 0 else 'planning'


def rate(k, n):
    lo, hi = wilson(k, n)
    return f'{k}/{n} = {100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%)' if n else '-'


def main(name):
    base = os.path.join(RES, name)
    eps = {}
    for m in METHODS:
        files = glob.glob(os.path.join(base, m, 'episodes', 'seed[0-9]*.json'))
        if files:
            eps[m] = {r['seed']: r for r in (json.load(open(f)) for f in files)}
    if not {'B2', 'b2_guided'} <= set(eps):
        print(f'{name}: B2 and b2_guided needed, have {sorted(eps)}')
        return
    dropped = sorted({s for m in eps for s, r in eps[m].items() if r.get('excluded_reason')})
    seeds = sorted(set.intersection(*[set(v) for v in eps.values()]) - set(dropped))
    groups = [('all', seeds), ('another route', [s for s in seeds if eps['B2'][s]['detour_cells'] is not None]),
              ('no other route', [s for s in seeds if eps['B2'][s]['detour_cells'] is None])]
    out = dict(name=name, methods=list(eps), n=len(seeds), dropped=dropped, by_group={})
    lines = [f'# {name}: B2 vs b2_guided (exploratory)', '',
             f'Paired scenarios: {len(seeds)} (dropped: {dropped or "none"})', '',
             '| Method | ' + ' | '.join(f'{g} (n={len(ss)})' for g, ss in groups) + ' | hit obstacle | planning time (s, mean) |',
             '|---|' + '---|' * (len(groups) + 2)]
    for m in eps:
        ks = [sum(eps[m][s]['success'] for s in ss) for _, ss in groups]
        hits = sum(eps[m][s]['obstacle_collision'] for s in seeds)
        t = sum(eps[m][s]['planning_time_s'] for s in seeds) / max(len(seeds), 1)
        out['by_group'][m] = {g: k for (g, _), k in zip(groups, ks)}
        lines.append(f'| {m} | ' + ' | '.join(rate(k, len(ss)) for k, (_, ss) in zip(ks, groups)) +
                     f' | {hits}/{len(seeds)} | {t:.0f} |')
    a = [eps['B2'][s]['success'] for s in seeds]
    b = [eps['b2_guided'][s]['success'] for s in seeds]
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    p = mcnemar_exact(only_a, only_b)
    out['mcnemar'] = dict(only_B2=only_a, only_b2_guided=only_b, p=p)
    lines += ['', f'Discordant pairs: only B2 {only_a}, only b2_guided {only_b}; exact McNemar two-sided p = {p:.4f}.', '',
              '## Where the obstacle collisions came from', '']
    for m in ('B2', 'b2_guided'):
        lay = [layer(eps[m][s]) for s in seeds]
        c = {k: lay.count(k) for k in ('planning', 'execution', 'no replan')}
        out.setdefault('collision_layer', {})[m] = c
        lines.append(f'- {m}: planning {c["planning"]}, execution {c["execution"]}, no replan before the hit {c["no replan"]}')
    lines += ['', '## Per scenario', '', '| seed | blocked | other route | ' + ' | '.join(eps) + ' |',
              '|---|---|---|' + '---|' * len(eps)]
    for s in seeds:
        r = eps['B2'][s]
        lines.append(f"| {s} | {r['blocked_cell']} | {'yes' if r['detour_cells'] is not None else 'no'} | " +
                     ' | '.join('ok' if eps[m][s]['success'] else eps[m][s]['failure_reason'] for m in eps) + ' |')
    with open(os.path.join(base, 'guided_loop.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(base, 'guided_loop.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    for n in sys.argv[1:] or ['partial_fresh', 'probe_partial']:
        main(n)
