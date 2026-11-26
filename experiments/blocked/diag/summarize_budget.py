"""
Summary and decision for budget_sampling.py, with the rules of experiment_progress/時間預算實驗計畫_2026-11-25.md
(E0: section 3; E1: section 4.3 as amended in section 10.3, before any E0/E1 data was seen;
E1r: section 11, written after E1 and before E1r).

    python experiments/blocked/diag/summarize_budget.py        (plain Python)
Writes results/diag/budget/summary.{md,json}.
"""
import glob
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BASE = os.path.join(REPO, 'experiments/blocked/results/diag/budget')
E1_GROUPS = ('S-small', 'S-large', 'L-small', 'L-large')


def rows(exp):
    return sorted((json.load(open(f)) for f in glob.glob(os.path.join(BASE, exp, 'rows', '*.json'))), key=lambda r: r['index'])


def pool(rs):
    n = sum(r['n'] for r in rs)
    k = {key: sum(r[f'n_{key}'] for r in rs) for key in ('intact', 'clear', 'valid', 'via_b')}
    broken = n - k['intact']
    through = sum(sum(s['intact'] and not s['clear'] for s in r['samples']) for r in rs)
    return dict(n_scen=len(rs), n=n, **k, broken=broken, intact_but_hit=through,
                scen_with_valid=sum(r['n_valid'] > 0 for r in rs))


def pct(k, n):
    return f'{k}/{n} = {100 * k / n:.1f}%' if n else '-'


def verdict(p):
    """'found' / 'not found' / 'neither' for one E1 group (section 4.3)."""
    if p['n'] == 0:
        return 'missing'
    if p['valid'] >= 0.05 * p['n'] and p['scen_with_valid'] >= 2:
        return 'found'
    if p['valid'] <= 0.01 * p['n'] and p['scen_with_valid'] <= 1:
        return 'not found'
    return 'neither'


def main():
    out, lines = {}, ['# Time budget or route rarity: E0 / E1 summary (exploratory)', '',
                      'Rules: experiment_progress/時間預算實驗計畫_2026-11-25.md (E0 section 3, E1 section 10.3).', '']
    e0 = rows('e0')
    if e0:
        groups = {g: pool([r for r in e0 if r['group'] == g]) for g in ('far', 'mid', 'control')}
        far, ctl = groups['far'], groups['control']
        f, c = far['intact'] / max(far['n'], 1), ctl['intact'] / max(ctl['n'], 1)
        if len(e0) < 9:
            d0 = f'incomplete ({len(e0)}/9 start cells)'
        elif f <= 0.25 and f < 0.5 * c:
            d0 = 'supports the time budget'
        elif abs(f - c) <= 0.10:
            d0 = 'does not support the time budget'
        else:
            d0 = 'uncertain'
        out['e0'] = dict(groups=groups, decision=d0)
        lines += ['## E0: no obstacle, start far from the goal', '',
                  '| start | group | BFS cells | intact | median arc length | median in-wall points |', '|---|---|---|---|---|---|']
        for r in e0:
            lines.append(f"| {tuple(r['A'])} | {r['group']} | {r['total']} | {pct(r['n_intact'], r['n'])} | "
                         f"{r['median_length']:.1f} | {r['median_in_wall']:.0f} |")
        lines += ['', '| group | intact |', '|---|---|'] + [f'| {g} | {pct(p["intact"], p["n"])} |' for g, p in groups.items()]
        lines += ['', f'**E0 decision: {d0}.** (far intact {100 * f:.1f}% vs control {100 * c:.1f}%)', '']

    e1 = rows('e1')
    if e1:
        groups = {g: pool([r for r in e1 if r['group'] == g]) for g in E1_GROUPS}
        v = {g: verdict(p) for g, p in groups.items()}
        s = [p for g, p in groups.items() if g.startswith('S-')]
        setup = sum(p['intact'] for p in s) / max(sum(p['n'] for p in s), 1)
        if len(e1) < 24:
            d1 = f'incomplete ({len(e1)}/24 scenarios)'
        elif setup < 0.5:
            d1 = f'uncertain (setup check failed: S intact {100 * setup:.0f}% < 50%)'
        elif v['S-large'] == 'found' and v['L-small'] == 'not found':
            d1 = 'supports the time budget'
        elif v['S-large'] == 'not found' and v['L-small'] == 'found':
            d1 = 'supports route rarity'
        elif v['S-small'] == 'found' and v['S-large'] == 'not found' and v['L-small'] == 'not found':
            d1 = 'both factors needed'
        elif v['S-small'] == 'not found' and v['S-large'] == 'not found' and v['L-small'] == 'not found':
            d1 = 'shortest-route preference dominates (any detour is not sampled)'
        else:
            d1 = 'uncertain'
        out['e1'] = dict(groups=groups, verdicts=v, setup_intact=setup, decision=d1)
        lines += ['## E1: total length x extra cells (one full obstacle)', '',
                  '| group | scenarios | valid | scenarios with valid | intact | broken | intact but hits obstacle | via blocked cell | verdict |',
                  '|---|---|---|---|---|---|---|---|---|']
        for g, p in groups.items():
            lines.append(f"| {g} | {p['n_scen']} | {pct(p['valid'], p['n'])} | {p['scen_with_valid']} | {pct(p['intact'], p['n'])} | "
                         f"{pct(p['broken'], p['n'])} | {pct(p['intact_but_hit'], p['n'])} | {pct(p['via_b'], p['n'])} | {v[g]} |")
        small = [r for r in e1 if r['group'] == 'S-small']
        for ex in (2, 4):
            p = pool([r for r in small if r['extra'] == ex])
            lines.append(f'\nS-small with extra {ex}: valid {pct(p["valid"], p["n"])} in {p["n_scen"]} scenarios.')
        lines += ['', f'Setup check: S groups intact {100 * setup:.0f}% (needs >= 50%).', '',
                  f'**E1 decision: {d1}.**', '', '### Per scenario', '',
                  '| id | group | A | G | b | base | total | extra | valid | intact | via b | median length |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in e1:
            lines.append(f"| {r['id']} | {r['group']} | {tuple(r['A'])} | {tuple(r['G'])} | {tuple(r['b'])} | {r['base']} | "
                         f"{r['total']} | {r['extra']} | {r['n_valid']}/{r['n']} | {r['n_intact']} | {r['n_via_b']} | {r['median_length']:.1f} |")
    e1r = rows('e1r')
    if e1r:
        lines += ['', '## E1r: replication with new scenarios (plan section 11)', '']
        out['e1r'] = {}
        for label, rs in (('E1r alone', e1r), ('E1 + E1r', [r for r in e1 if r['group'] != 'L-large'] + e1r)):
            groups = {g: pool([r for r in rs if r['group'] == g]) for g in E1_GROUPS[:3]}
            v = {g: verdict(p) for g, p in groups.items()}
            s = [p for g, p in groups.items() if g.startswith('S-')]
            setup = sum(p['intact'] for p in s) / max(sum(p['n'] for p in s), 1)
            if setup < 0.5:
                d = f'uncertain (setup check failed: S intact {100 * setup:.0f}% < 50%)'
            elif v['S-large'] == 'found' and v['L-small'] == 'not found':
                d = 'supports the time budget'
            elif v['S-large'] == 'not found' and v['L-small'] == 'found':
                d = 'supports route rarity'
            elif all(v[g] == 'not found' for g in E1_GROUPS[:3]):
                d = 'shortest-route preference dominates (any detour is not sampled)'
            else:
                d = 'uncertain'
            out['e1r'][label] = dict(groups=groups, verdicts=v, setup_intact=setup, decision=d)
            lines += [f'### {label}', '', '| group | scenarios | valid | scenarios with valid | intact | via blocked cell | verdict |',
                      '|---|---|---|---|---|---|---|']
            for g, p in groups.items():
                lines.append(f"| {g} | {p['n_scen']} | {pct(p['valid'], p['n'])} | {p['scen_with_valid']} | {pct(p['intact'], p['n'])} | "
                             f"{pct(p['via_b'], p['n'])} | {v[g]} |")
            lines += ['', f'**{label} decision: {d}.**', '']
        lines += ['### E1r per scenario', '', '| id | group | A | G | b | base | total | extra | valid | intact | via b | median length |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in e1r:
            lines.append(f"| {r['id']} | {r['group']} | {tuple(r['A'])} | {tuple(r['G'])} | {tuple(r['b'])} | {r['base']} | "
                         f"{r['total']} | {r['extra']} | {r['n_valid']}/{r['n']} | {r['n_intact']} | {r['n_via_b']} | {r['median_length']:.1f} |")
    with open(os.path.join(BASE, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(BASE, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
