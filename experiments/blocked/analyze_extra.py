"""
Paired analysis of the 11/23 follow-up experiments (final_project/補強實驗計畫_2026-11-23.md; exploratory).

    python experiments/blocked/analyze_extra.py            (plain Python)

  partial_exec : experiment 2, B2 / b2_guided with the original executor and with executor "path", plus the oracle
  full_fresh   : experiment 3, B0 B1 B2 oracle and B2 with lag_replan (full_fresh_lag)

Seeds excluded in any method are dropped from all. The last group of each study is 'all'. Writes results/<name>/extra.{md,json}.
"""
import glob
import json
import os
from collections import Counter

from analyze_guided_loop import layer, rate
from analyze_main import RES, mcnemar_exact

STUDIES = {
    'partial_exec': dict(
        runs=[('oracle', 'partial_exec', 'oracle'), ('B2', 'partial_exec', 'B2'),
              ('b2_guided', 'partial_exec', 'b2_guided'), ('B2 + path', 'partial_exec_path', 'B2'),
              ('b2_guided + path', 'partial_exec_path', 'b2_guided')],
        groups=lambda r: [('another route', r['detour_cells'] is not None),
                          ('no other route', r['detour_cells'] is None), ('all', True)],
        pairs=[('b2_guided', 'b2_guided + path', 'all'), ('B2', 'B2 + path', 'all'),
               ('B2 + path', 'b2_guided + path', 'all'), ('B2', 'b2_guided', 'all')]),
    'full_fresh': dict(
        runs=[('B0', 'full_fresh', 'B0'), ('B1', 'full_fresh', 'B1'), ('B2', 'full_fresh', 'B2'),
              ('B2_lag', 'full_fresh_lag', 'B2'), ('oracle', 'full_fresh', 'oracle')],
        groups=lambda r: [('longer detour', r['detour_cells'] > r['base_cells']),
                          ('same length', r['detour_cells'] == r['base_cells']), ('all', True)],
        pairs=[('B2', 'B2_lag', 'longer detour'), ('B1', 'B2_lag', 'longer detour'), ('B1', 'B2', 'longer detour'),
               ('B2', 'B2_lag', 'all')]),
}


def load(run, method):
    files = glob.glob(os.path.join(RES, run, method, 'episodes', 'seed[0-9]*.json'))
    return {r['seed']: r for r in (json.load(open(f)) for f in files)}


def main(name, study):
    eps = {lab: load(run, m) for lab, run, m in study['runs']}
    eps = {k: v for k, v in eps.items() if v}
    if len(eps) < 2:
        print(f'{name}: not enough runs yet ({sorted(eps)})')
        return
    dropped = sorted({s for v in eps.values() for s, r in v.items() if r.get('excluded_reason')})
    seeds = sorted(set.intersection(*[set(v) for v in eps.values()]) - set(dropped))
    ref = next(iter(eps.values()))
    gnames = [g for g, _ in study['groups'](ref[seeds[0]])]
    members = {g: [s for s in seeds if dict(study['groups'](ref[s]))[g]] for g in gnames}
    out = dict(name=name, methods=list(eps), n=len(seeds), dropped=dropped, success={}, pairs=[], collision_layer={},
               failures={})
    lines = [f'# {name}: follow-up experiment (exploratory, 2026-11-23)', '',
             f'Paired scenarios: {len(seeds)} (dropped: {dropped or "none"})', '',
             '| Method | ' + ' | '.join(f'{g} (n={len(members[g])})' for g in gnames) +
             ' | hit obstacle | planning time (s, mean) |', '|---|' + '---|' * (len(gnames) + 2)]
    for m in eps:
        ks = {g: sum(eps[m][s]['success'] for s in members[g]) for g in gnames}
        out['success'][m] = ks
        hits = sum(eps[m][s]['obstacle_collision'] for s in seeds)
        t = sum(eps[m][s]['planning_time_s'] for s in seeds) / max(len(seeds), 1)
        lines.append(f'| {m} | ' + ' | '.join(rate(ks[g], len(members[g])) for g in gnames) + f' | {hits}/{len(seeds)} | {t:.0f} |')
    lines += ['', '## Paired comparisons (exact McNemar, two-sided)', '']
    for a, b, g in study['pairs']:
        if a not in eps or b not in eps:
            continue
        ss = members[g]
        oa = sum(eps[a][s]['success'] and not eps[b][s]['success'] for s in ss)
        ob_ = sum(eps[b][s]['success'] and not eps[a][s]['success'] for s in ss)
        p = mcnemar_exact(oa, ob_)
        out['pairs'].append(dict(a=a, b=b, group=g, n=len(ss), only_a=oa, only_b=ob_, p=p))
        lines.append(f'- {a} vs {b} ({g}, n={len(ss)}): only {a} {oa}, only {b} {ob_}; p = {p:.4f}')
    lines += ['', '## Failure reasons and where obstacle collisions came from', '']
    for m in eps:
        f = Counter(eps[m][s]['failure_reason'] for s in seeds if not eps[m][s]['success'])
        lay = Counter(layer(eps[m][s]) for s in seeds if eps[m][s]['obstacle_collision'])
        nl = sum(eps[m][s].get('n_lag_replans', 0) for s in seeds)
        out['failures'][m], out['collision_layer'][m] = dict(f), dict(lay)
        lines.append(f'- {m}: failures {dict(f) or "none"}; collisions by layer {dict(lay) or "none"}' +
                     (f'; lag replans {nl}' if nl else ''))
    lines += ['', '## Per scenario', '', '| seed | blocked | ' + ' | '.join(gnames[:-1]) + ' | ' + ' | '.join(eps) + ' |',
              '|---|---|' + '---|' * (len(gnames) - 1 + len(eps))]
    for s in seeds:
        r = ref[s]
        flags = dict(study['groups'](r))
        lines.append(f"| {s} | {r['blocked_cell']} | " + ' | '.join('x' if flags[g] else '' for g in gnames[:-1]) + ' | ' +
                     ' | '.join('ok' if eps[m][s]['success'] else eps[m][s]['failure_reason'] for m in eps) + ' |')
    with open(os.path.join(RES, name, 'extra.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES, name, 'extra.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    for n, st in STUDIES.items():
        if os.path.isdir(os.path.join(RES, n)):
            main(n, st)
