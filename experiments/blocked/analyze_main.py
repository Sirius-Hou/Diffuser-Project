"""
The pre-registered analysis of main (PREREGISTRATION_main.md), over the three runs that share its scenarios.

    python experiments/blocked/analyze_main.py            (plain Python, no numpy needed)

Reads results/main/<method>, results/main_junctions/ours and results/main_lag_line/astar_line; writes
results/main_primary.md and results/main_primary.json. Methods that have not run yet are skipped.
Seeds excluded in any method are dropped from every method, as in analyze_blocked.py.
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
RUNS = [('B0', 'main', 'B0'), ('B1', 'main', 'B1'), ('B2', 'main', 'B2'), ('ours (corners)', 'main', 'ours'),
        ('ours (junctions)', 'main_junctions', 'ours'), ('astar_line + lag', 'main_lag_line', 'astar_line'),
        ('oracle', 'main', 'oracle')]
PRIMARY = ('B2', 'ours (junctions)')
RADII = (0.5, 0.75, 1.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def rate(k, n):
    lo, hi = wilson(k, n)
    return f'{k}/{n} = {100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%)' if n else '-'


def main():
    scen = {s['seed']: s for s in json.load(open(os.path.join(RES, 'main', 'scenarios.json')))['scenarios']}
    eps = {}
    for name, run, method in RUNS:
        files = sorted(glob.glob(os.path.join(RES, run, method, 'episodes', 'seed[0-9]*.json')))
        if files:
            eps[name] = {r['seed']: r for r in (json.load(open(f)) for f in files)}
    names = [n for n, _, _ in RUNS if n in eps]
    dropped = sorted({s for n in names for s, r in eps[n].items() if r.get('excluded_reason')})
    seeds = sorted(set(scen).intersection(*[set(eps[n]) for n in names]) - set(dropped))

    def extra(s):
        return scen[s]['detour_cells'] - scen[s]['base_cells']
    longer = [s for s in seeds if extra(s) > 0]
    same = [s for s in seeds if extra(s) == 0]
    out = dict(methods=names, n_scenarios=len(seeds), dropped_seeds=dropped,
               n_longer=len(longer), n_same=len(same))
    lines = [f'# main: pre-registered analysis', '',
             f'Scenarios: {len(seeds)} paired (dropped: {dropped or "none"}); '
             f'{len(longer)} with a longer detour, {len(same)} with a same-length route.', '']

    if all(p in eps for p in PRIMARY):
        a, b_ = PRIMARY
        sa = [eps[a][s]['success'] for s in longer]
        sb = [eps[b_][s]['success'] for s in longer]
        only_a = sum(x and not y for x, y in zip(sa, sb))
        only_b = sum(y and not x for x, y in zip(sa, sb))
        p = mcnemar_exact(only_a, only_b)
        out['primary'] = dict(population='full, longer detour', n=len(longer), success={a: sum(sa), b_: sum(sb)},
                              only_first=only_a, only_second=only_b, both=sum(x and y for x, y in zip(sa, sb)),
                              neither=sum(not x and not y for x, y in zip(sa, sb)), mcnemar_exact_p=p)
        lines += ['## Primary: B2 vs ours (junctions), full blockage with a longer detour', '',
                  '| | n | Success (Wilson 95% CI) |', '|---|---|---|',
                  f'| {a} | {len(longer)} | {rate(sum(sa), len(longer))} |',
                  f'| {b_} | {len(longer)} | {rate(sum(sb), len(longer))} |', '',
                  f'Discordant pairs: only {a} succeeds {only_a}, only {b_} succeeds {only_b}; '
                  f'both {out["primary"]["both"]}, neither {out["primary"]["neither"]}. '
                  f'Exact McNemar two-sided p = {p:.4f}.', '']

    lines += ['## Secondary 1 and 3: every method by case', '',
              '| Method | Longer detour | Same length | All |', '|---|---|---|---|']
    out['by_case'] = {}
    for n in names:
        k = {grp: sum(eps[n][s]['success'] for s in ss) for grp, ss in (('longer', longer), ('same', same), ('all', seeds))}
        out['by_case'][n] = k
        lines.append(f"| {n} | {rate(k['longer'], len(longer))} | {rate(k['same'], len(same))} | {rate(k['all'], len(seeds))} |")

    lines += ['', '## Secondary 2: success by extra detour cells (longer detours only)', '']
    bins = [('<= 6', [s for s in longer if extra(s) <= 6]), ('>= 8', [s for s in longer if extra(s) >= 8])]
    lines += ['| Method | ' + ' | '.join(f'extra {b} (n={len(ss)})' for b, ss in bins) + ' |',
              '|---|' + '---|' * len(bins)]
    out['by_extra'] = {}
    for n in names:
        out['by_extra'][n] = {b: sum(eps[n][s]['success'] for s in ss) for b, ss in bins}
        lines.append(f'| {n} | ' + ' | '.join(rate(out['by_extra'][n][b], len(ss)) for b, ss in bins) + ' |')

    lines += ['', '## Secondary 4: success radius (longer detours; min distance to the goal over the episode)', '',
              '| Method | ' + ' | '.join(f'r = {r}' for r in RADII) + ' |', '|---|' + '---|' * len(RADII)]
    out['by_radius'] = {}
    for n in names:
        out['by_radius'][n] = {str(r): sum(eps[n][s]['min_dist_to_goal'] <= r for s in longer) for r in RADII}
        lines.append(f'| {n} | ' + ' | '.join(rate(out['by_radius'][n][str(r)], len(longer)) for r in RADII) + ' |')

    lines += ['', '## Secondary 5: A* use and blocked cells (longer detours)', '']
    cells = {}
    for s in longer:
        c = tuple(scen[s]['obstacle']['cell'])
        cells[c] = cells.get(c, 0) + 1
    lines.append('Blocked cells: ' + ', '.join(f'{c}: {k}' for c, k in sorted(cells.items(), key=lambda x: -x[1])))
    out['blocked_cells'] = {str(c): k for c, k in cells.items()}
    for n in names:
        used = [s for s in longer if eps[n][s].get('used_astar')]
        if used:
            k = sum(eps[n][s]['success'] for s in used)
            lines.append(f'- {n}: A* used in {len(used)}/{len(longer)}, success {k}/{len(used)} in those')
            out.setdefault('astar_used', {})[n] = dict(n=len(used), success=k)

    with open(os.path.join(RES, 'main_primary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES, 'main_primary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
