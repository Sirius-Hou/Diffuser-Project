"""
Closed-loop evidence for experiment_progress/時間預算實驗計畫_2026-11-25.md, section 10 (post hoc, exploratory).

  1. Success by the route left when the obstacle appears. The route is measured from the ball's cell at t_appear
     (B0's rollout; every method follows the same plan before t_appear), not from scenarios.json's cell_at_appear,
     which is where the t=0 plan is at t_appear (the ball lags behind it).
  2. ours' legs: length, speed, and how often a leg's samples clear the obstacle, split by whether the shortest
     route from the leg's start to its waypoint is blocked (the leg itself needs a detour).

    python experiments/blocked/analyze_budget_closed_loop.py          (plain Python + numpy)
Writes results/diag/budget/closed_loop.{md,json}.
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from analyze_main import RES, wilson  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402

LARGE_MAZE = ['############', '#OOOO#OOOOO#', '#O##O#O#O#O#', '#OOOOOO#OOO#', '#O####O###O#', '#OO#O#OOOOO#',
              '##O#O#O#O###', '#OO#OOO#OGO#', '############']          # d4rl pointmaze LARGE_MAZE
MAZE = np.array([[mu.WALL if ch == '#' else mu.EMPTY for ch in row] for row in LARGE_MAZE])
SETS = {'main': [('B0', 'main/B0'), ('B1', 'main/B1'), ('B2', 'main/B2'), ('ours (corners)', 'main/ours'),
                 ('ours (junctions)', 'main_junctions/ours'), ('astar_line + lag', 'main_lag_line/astar_line'),
                 ('oracle', 'main/oracle')],
        'full_fresh': [('B0', 'full_fresh/B0'), ('B1', 'full_fresh/B1'), ('B2', 'full_fresh/B2'),
                       ('B2 + lag replan', 'full_fresh_lag/B2'), ('oracle', 'full_fresh/oracle')]}
BINS = ((0, 12), (13, 15), (16, 18), (19, 99))
LEGS = [('ours (corners)', 'main/ours'), ('ours (junctions)', 'main_junctions/ours')]


def cells(a, g, blocked=()):
    path = mu.bfs_path(MAZE, tuple(a), tuple(g), blocked)
    return None if path is None else len(path) - 1


def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda k: math.comb(r1, k) * math.comb(n - r1, c1 - k) / math.comb(n, c1)
    p0 = p(a)
    return sum(p(k) for k in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(k) <= p0 * (1 + 1e-9))


def load(d):
    return {r['seed']: r for r in (json.load(open(f)) for f in glob.glob(os.path.join(RES, d, 'episodes', 'seed[0-9]*.json')))}


def remaining(name):
    scen = {s['seed']: s for s in json.load(open(os.path.join(RES, name, 'scenarios.json')))['scenarios']}
    eps = {m: load(d) for m, d in SETS[name]}
    dropped = sorted({s for m in eps for s, r in eps[m].items() if r.get('excluded_reason')})
    seeds = sorted(set.intersection(*[set(v) for v in eps.values()]) - set(dropped))
    b0 = dict(SETS[name])['B0']
    rows = []
    for s in seeds:
        sc = scen[s]
        if sc['obstacle']['kind'] != 'full' or sc['detour_cells'] is None:
            continue
        ball = mu.xy_to_cell(np.load(os.path.join(RES, b0, 'episodes', f'seed{s}.npz'))['rollout_xy'][sc['t_appear']])
        tot = cells(ball, sc['goal_cell'], [tuple(sc['obstacle']['cell'])])
        base = cells(ball, sc['goal_cell'])
        rows.append(dict(seed=s, ball_cell=list(map(int, ball)), plan_cell=sc['cell_at_appear'], total=tot, extra=tot - base,
                         success={m: bool(eps[m][s]['success']) for m in eps},
                         failure={m: eps[m][s]['failure_reason'] for m in eps}))
    return dict(dropped=dropped, methods=[m for m, _ in SETS[name]], rows=rows)


def legs(d, scen):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, d, 'episodes', 'seed[0-9]*.json'))):
        r = json.load(open(f))
        xy = np.load(f[:-5] + '.npz')['rollout_xy']
        b = tuple(scen[r['seed']]['obstacle']['cell'])
        starts = {}
        for e in r['events']:
            if e['event'] != 'leg_plan':
                continue
            c = mu.xy_to_cell(xy[min(e['t'], len(xy) - 1)])
            free, blk = cells(c, e['waypoint']), cells(c, e['waypoint'], [b])
            starts.setdefault(e['leg'], e['t'])
            out.append(dict(seed=r['seed'], leg=e['leg'], t=e['t'], why=e['why'], cells=e['cells'], horizon=e['horizon'],
                            n=e['n'], n_clear=e['n_clear'], leg_shortest_blocked=bool(blk is None or blk > free)))
        reached = {e['leg']: e['t'] for e in r['events'] if e['event'] == 'waypoint_reached'}
        for x in out:
            if x['seed'] == r['seed'] and x['leg'] in reached and x['t'] == starts[x['leg']]:
                x['steps_per_cell'] = (reached[x['leg']] - x['t']) / max(x['cells'], 1)
    return out


def rate(k, n):
    lo, hi = wilson(k, n)
    return f'{k}/{n}' if n else '-'


def main():
    out = {'remaining': {n: remaining(n) for n in SETS}}
    lines = ['# Closed-loop evidence for the time-budget question (post hoc, exploratory)', '',
             'Route left = BFS cells from the ball\'s cell at t_appear (B0 rollout) to the goal with the obstacle cell blocked.', '']
    for name, d in out['remaining'].items():
        for pop, sel in (('longer detour', lambda r: r['extra'] > 0), ('all', lambda r: True)):
            rs = [r for r in d['rows'] if sel(r)]
            lines += [f'## {name}: {pop} (n = {len(rs)}; dropped {d["dropped"] or "none"})', '',
                      '| method | ' + ' | '.join(f'{lo}-{hi}' if hi < 99 else f'>= {lo}' for lo, hi in BINS) +
                      ' | <= 15 vs >= 16 | Fisher p |', '|---|' + '---|' * (len(BINS) + 2)]
            for m in d['methods']:
                cells_ = []
                for lo, hi in BINS:
                    b = [r for r in rs if lo <= r['total'] <= hi]
                    cells_.append(rate(sum(r['success'][m] for r in b), len(b)))
                A = [r for r in rs if r['total'] <= 15]
                B = [r for r in rs if r['total'] >= 16]
                a, c = sum(r['success'][m] for r in A), sum(r['success'][m] for r in B)
                lines.append(f'| {m} | ' + ' | '.join(cells_) + f' | {a}/{len(A)} vs {c}/{len(B)} | '
                             f'{fisher(a, len(A) - a, c, len(B) - c):.3f} |')
            lines.append('')
    pooled = [r for d in out['remaining'].values() for r in d['rows'] if r['extra'] > 0]
    lines += ['## Can the closed loop separate total length from extra cells? (main + full_fresh, longer detours, B2)', '',
              '| | extra <= 6 | extra >= 8 |', '|---|---|---|']
    for T, tsel in (('total <= 15', lambda r: r['total'] <= 15), ('total >= 16', lambda r: r['total'] >= 16)):
        row = []
        for E, esel in (('<= 6', lambda r: r['extra'] <= 6), ('>= 8', lambda r: r['extra'] >= 8)):
            rs = [r for r in pooled if tsel(r) and esel(r)]
            row.append(rate(sum(r['success']['B2'] for r in rs), len(rs)))
        lines.append(f'| {T} | ' + ' | '.join(row) + ' |')
    scen = {s['seed']: s for s in json.load(open(os.path.join(RES, 'main', 'scenarios.json')))['scenarios']}
    out['legs'] = {}
    lines += ['', '## ours: legs', '']
    for name, d in LEGS:
        L = legs(d, scen)
        out['legs'][name] = L
        spc = [x['steps_per_cell'] for x in L if 'steps_per_cell' in x]
        lines += [f'### {name}: {len(L)} leg plans', '',
                  f'- leg length (cells): max {max(x["cells"] for x in L)}, horizon max {max(x["horizon"] for x in L)}',
                  f'- steps per cell to reach the waypoint: median {np.median(spc):.1f} (p10 {np.percentile(spc, 10):.1f}, '
                  f'p90 {np.percentile(spc, 90):.1f}; n = {len(spc)})', '',
                  '| leg | plans | scenarios | samples clear |', '|---|---|---|---|']
        for lab, v in (('shortest route to the waypoint open', False), ('shortest route to the waypoint blocked', True)):
            xs = [x for x in L if x['leg_shortest_blocked'] == v]
            if xs:
                k, n = sum(x['n_clear'] for x in xs), sum(x['n'] for x in xs)
                lines.append(f'| {lab} | {len(xs)} | {len({x["seed"] for x in xs})} | {k}/{n} = {100 * k / n:.1f}% |')
        lines.append('')
    base = os.path.join(RES, 'diag', 'budget')
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, 'closed_loop.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(base, 'closed_loop.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
