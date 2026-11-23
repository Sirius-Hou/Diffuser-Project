"""
Tables and the decision of the guidance diagnostic (Guidance實驗計畫_2026-11-22.md, section 10).

    bash experiments/common/setup/run.sh python experiments/blocked/diag/summarize_guided.py

G0 (full) is diagnostic #2 at H = 384 (trigger_sampling, identical sampler output at scale 0), relabelled from
its saved plans with the same labels as guided_sampling.py. Writes results/diag/guided_sampling/summary.{md,json}.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common.planner import load_env  # noqa: E402
from guided_sampling import label, scenario_meta, scenarios  # noqa: E402

RES = os.path.join(REPO, 'experiments/blocked/results')
OUT = os.path.join(RES, 'diag/guided_sampling')
MARGIN = 0.05
A_MIN_SCEN, A_MIN_PLANS = 2, 4


def g0_full(maze):
    rows = []
    for name, sc, _ in scenarios('full', ['probe', 'probe_multigoal']):
        plans = np.load(os.path.join(RES, 'diag/trigger_sampling', f'{name}_seed{sc["seed"]:03d}_H384.npy'))
        obstacle = ob.Obstacle.from_dict(sc['obstacle'])
        detour_only, axis = scenario_meta(maze, sc)
        labels = [label(maze, p, obstacle, MARGIN, detour_only, axis) for p in plans]
        rows.append(dict(results=name, seed=sc['seed'], base_cells=sc['base_cells'], detour_cells=sc['detour_cells'],
                         n=len(labels), samples=labels))
    return dict(config=dict(tag='full_G0', scale=0.0), rows=rows)


def stats(rows):
    S = [s for r in rows for s in r['samples']]
    n = len(S)
    fail = {k: sum(s['fail'] == k for s in S) for k in ('still_hit', 'split', 'wall')}
    return dict(n=n, valid=sum(s['valid'] for s in S),
                valid_via=sum(bool(s['valid'] and s['via_detour']) for s in S),
                fail=fail, median_in_wall=float(np.median([s['in_wall'] for s in S])) if S else None)


def main():
    maze = load_env('maze2d-large-v1').maze_arr
    groups = {'full_G0': g0_full(maze)}
    for d in sorted(glob.glob(os.path.join(OUT, '*', 'rows.json'))):
        tag = os.path.basename(os.path.dirname(d))
        if tag.startswith(('full_', 'partial_')):
            groups[tag] = json.load(open(d))
    out, lines = {}, ['# Guidance diagnostic (offline, H = 384, 64 plans per scenario)', '']

    full = {t: g for t, g in groups.items() if t.startswith('full_')}
    lines += ['## E1: full blockage (22 scenarios with a longer detour)', '',
              '| Group | scale | cost | extra <= 6: valid | extra >= 8: valid (via detour) | >= 8 scenarios with >= 4 valid+via '
              '| >= 8 fail: still_hit / split / wall | median in-wall |',
              '|---|---|---|---|---|---|---|---|']
    for t, g in full.items():
        c = g['config']
        rows = g['rows']
        short = [r for r in rows if r['detour_cells'] - r['base_cells'] <= 6]
        long_ = [r for r in rows if r['detour_cells'] - r['base_cells'] >= 8]
        s_short, s_long = stats(short), stats(long_)
        a_count = sum(sum(bool(s['valid'] and s['via_detour']) for s in r['samples']) >= A_MIN_PLANS for r in long_)
        cost = ('walls+smooth' if c.get('walls') else 'obstacle') + (f" ({c.get('variant', 'x0')})" if c.get('scale') else '')
        out[t] = dict(config=c, n_scen=len(rows), short=s_short, long=s_long, long_scen_a=a_count)
        f = s_long['fail']
        lines.append(f"| {t} | {c.get('scale', 0)} | {cost} | {s_short['valid']}/{s_short['n']} | "
                     f"{s_long['valid']}/{s_long['n']} ({s_long['valid_via']}) | {a_count}/{len(long_)} | "
                     f"{f['still_hit']} / {f['split']} / {f['wall']} | {s_long['median_in_wall']:.0f} |")
        if len(rows) != 22:
            lines[-1] += f' (only {len(rows)} scenarios so far)'

    part = {t: g for t, g in groups.items() if t.startswith('partial_')}
    if part:
        lines += ['', '## E2: partial blockage, gap 0.5 (probe_partial, 8 scenarios)', '',
                  '| Group | scale | cost | other route: valid | no other route: valid | no-route scenarios above P0 '
                  '| median gap offset of valid plans |', '|---|---|---|---|---|---|---|']
        p0 = {r['seed']: r['n_valid'] for r in part.get('partial_P0', {'rows': []})['rows']}
        for t, g in part.items():
            c = g['config']
            rows = g['rows']
            other = [r for r in rows if r['detour_cells'] is not None]
            none_ = [r for r in rows if r['detour_cells'] is None]
            so, sn = stats(other), stats(none_)
            above = sum(r['n_valid'] > p0.get(r['seed'], 0) for r in none_) if t != 'partial_P0' else '-'
            gaps = [s['gap_dev'] for r in rows for s in r['samples'] if s['valid'] and s.get('gap_dev') is not None]
            cost = 'walls+smooth' if c.get('walls') else 'obstacle'
            out[t] = dict(config=c, other=so, none=sn, none_above_p0=above,
                          median_gap_dev=float(np.median(gaps)) if gaps else None)
            lines.append(f"| {t} | {c.get('scale', 0)} | {cost} | {so['valid']}/{so['n']} | {sn['valid']}/{sn['n']} "
                         f"({100 * sn['valid'] / max(sn['n'], 1):.0f}%) | {above}/{len(none_)} | "
                         f"{np.median(gaps) if gaps else float('nan'):.3f} |")

    g0_short = out['full_G0']['short']['valid']
    a = [t for t, v in out.items() if t.startswith('full_') and t != 'full_G0' and v['long_scen_a'] >= A_MIN_SCEN]
    cc = [t for t, v in out.items() if t.startswith('full_') and t != 'full_G0' and v['short']['valid'] >= 2 * g0_short]
    lines += ['', '## Decision (section 10)', '',
              f'- A (another corridor): groups with >= {A_MIN_SCEN} long-detour scenarios having >= {A_MIN_PLANS}/64 '
              f'valid plans through the detour: {a or "none"}',
              f'- C (short detours easier): groups with extra <= 6 valid >= 2 x G0 ({2 * g0_short}): {cc or "none"}',
              f'- => {"run E3 (closed loop)" if a or cc else "no closed loop; negative result"}']
    out['decision'] = dict(A=a, C=cc)
    out['G0_labels'] = {f"{r['results']}_{r['seed']}": [s['fail'] or 'valid' for s in r['samples']]
                        for r in groups['full_G0']['rows']}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
