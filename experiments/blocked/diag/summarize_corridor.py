"""
Summary and decisions for F1-F7 of experiment_progress/走廊偏好實驗計畫_2026-11-27.md (sections 3, 6, 7, 8).

    python experiments/blocked/diag/summarize_corridor.py        (plain Python)
Writes results/diag/corridor/summary.{md,json}.
"""
import glob
import json
import os
import random
import statistics as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BASE = os.path.join(REPO, 'experiments/blocked/results/diag/corridor')
N_PERM = 10000
PERM_SEED = 20261127


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return st.correlation(ranks(x), ranks(y))


def perm_p(x, y):
    rng = random.Random(PERM_SEED)
    r0, rx, ry = spearman(x, y), ranks(x), ranks(y)
    hits = 0
    for _ in range(N_PERM):
        ry_ = ry[:]
        rng.shuffle(ry_)
        hits += abs(st.correlation(rx, ry_)) >= abs(r0) - 1e-12
    return r0, (hits + 1) / (N_PERM + 1)


def fmt(v):
    return '-' if v is None else f'{v:.2f}'


def main():
    out, lines = {}, ['# Corridor preference: F1-F4 summary (exploratory)', '',
                      'Rules: experiment_progress/走廊偏好實驗計畫_2026-11-27.md, section 3. y = share of a scenario\'s 64 '
                      'E1/E1r samples that do not pass the blocked cell b (the Diffuser is never told about b).', '']
    geo = json.load(open(os.path.join(BASE, 'geometry.json')))
    ids = [g['id'] for g in geo]
    y = [g['y'] for g in geo]

    # F1
    delta = [g['delta'] for g in geo]
    r1, p1 = perm_p(delta, y)
    d1 = 'supports the straight-line hypothesis' if r1 <= -0.5 and p1 < 0.05 else \
        'does not support the straight-line hypothesis' if r1 > -0.25 else 'uncertain'
    big = [(g['delta'] < 0) == (g['y'] > 0.5) for g in geo if abs(g['delta']) >= 0.2]
    out['F1'] = dict(rho=r1, p=p1, decision=d1, sign_agree=[sum(big), len(big)])
    lines += ['## F1: distance of the two routes to the straight line A-G', '',
              '| id | group | base | total | d(R_b) | d(R_d) | delta | y |', '|---|---|---|---|---|---|---|---|']
    lines += [f"| {g['id']} | {g['group']} | {g['base']} | {g['total']} | {g['d_line_b']:.2f} | {g['d_line_d']:.2f} | "
              f"{g['delta']:+.2f} | {g['y']:.2f} |" for g in geo]
    lines += ['', f'Spearman rho(delta, y) = {r1:.2f}, permutation p = {p1:.3f}; |delta| >= 0.2: sign agrees in '
              f'{sum(big)}/{len(big)}.', f'**F1 decision: {d1}.**', '']

    # F2
    e2 = {}
    for e in ('e1', 'e1r'):
        e2.update({r['id']: r for r in json.load(open(os.path.join(BASE, 'e2_all', f'summary_{e}.json')))['rows']})
    x = [e2[i]['n_avoid_b'] / e2[i]['n_segments'] for i in ids]
    r2, p2 = perm_p(x, y)
    d2 = 'supports data frequency' if r2 >= 0.5 and p2 < 0.05 else 'does not support data frequency' if r2 < 0.25 else 'uncertain'
    rep = [k for k, i in enumerate(ids) if i.startswith('e1r')]
    r2r, p2r = perm_p([x[k] for k in rep], [y[k] for k in rep])
    out['F2'] = dict(rho=r2, p=p2, decision=d2, e1r_rho=r2r, e1r_p=p2r)
    lines += ['## F2: share of training segments A -> G that avoid b', '',
              '| id | segments | avoiding b | x | segments <= 384 steps | avoiding b <= 384 | y |', '|---|---|---|---|---|---|---|']
    lines += [f"| {i} | {e2[i]['n_segments']} | {e2[i]['n_avoid_b']} | {xv:.2f} | {e2[i]['all_le_H']} | {e2[i]['avoid_le_H']} | "
              f"{yv:.2f} |" for i, xv, yv in zip(ids, x, y)]
    lines += ['', f'Spearman rho(x, y) = {r2:.2f}, permutation p = {p2:.3f} (24 scenarios); E1r alone (not seen before): '
              f'rho = {r2r:.2f}, p = {p2r:.3f}.', f'**F2 decision: {d2}.**', '']

    # F3
    tr = {r['id']: r for r in (json.load(open(f)) for f in glob.glob(os.path.join(BASE, 'trace', '*.json')))}
    if tr:
        rows = [tr[i] for i in ids if i in tr]
        locks = [v for r in rows for v in r['lock']]
        med = st.median(locks)
        d3 = 'decided early' if med <= 64 else 'decided late' if med >= 192 else 'decided in the middle'
        premise = [r['d_line'][0] < min(r['d_b'][0], r['d_d'][0]) for r in rows]
        agree = {j: sum(r['agree'][str(j)] * r['n'] for r in rows) / sum(r['n'] for r in rows) for j in (0, 16, 32, 64, 128)}
        repro = sum(r['reproduced'] for r in rows)
        out['F3'] = dict(n_scen=len(rows), reproduced=repro, lock_median=med, decision=d3,
                         premise=[sum(premise), len(premise)], agree=agree,
                         lock_quantiles=[st.quantiles(locks, n=10)[k] for k in (0, 4, 8)])
        lines += ['## F3: the model\'s clean-trajectory prediction during denoising (step 0 = first, noisiest)', '',
                  f'Reproduced E1/E1r exactly (per-sample via_b and valid): {repro}/{len(rows)} scenarios.', '',
                  '| id | y | lock median | lock p90 | agree j=0 | j=16 | j=64 | D_line / D_b / D_d at j=0 | at j=255 | spread j=0 / 64 / 255 |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            q = sorted(r['lock'])
            lines.append(f"| {r['id']} | {1 - sum(r['final_via_b']) / r['n']:.2f} | {st.median(q):.0f} | {q[int(0.9 * (len(q) - 1))]} | "
                         f"{r['agree']['0']:.2f} | {r['agree']['16']:.2f} | {r['agree']['64']:.2f} | "
                         f"{r['d_line'][0]:.2f} / {r['d_b'][0]:.2f} / {r['d_d'][0]:.2f} | "
                         f"{r['d_line'][-1]:.2f} / {r['d_b'][-1]:.2f} / {r['d_d'][-1]:.2f} | "
                         f"{r['spread'][0]:.2f} / {r['spread'][64]:.2f} / {r['spread'][-1]:.2f} |")
        lines += ['', f'Lock step over {len(locks)} samples: median {med:.0f}, p10 / p50 / p90 = '
                  + ' / '.join(f'{v:.0f}' for v in out['F3']['lock_quantiles']) + ' (of 256).',
                  'Agreement of via_b at step j with the final plan: ' + ', '.join(f'j={j}: {a:.2f}' for j, a in agree.items()) + '.',
                  f'Straight-line premise (at j = 0, D_line < min(D_b, D_d)): {sum(premise)}/{len(premise)} scenarios '
                  f'({"holds" if sum(premise) >= 2 * len(premise) / 3 else "does not hold"}).',
                  f'**F3 decision: {d3}.**', '']

    # F4
    ini = {r['id']: r for r in (json.load(open(f)) for f in glob.glob(os.path.join(BASE, 'init', '*.json')))}
    if ini:
        rows = [ini[i] for i in ids if i in ini]
        base_n = sum(g['n'] for g in geo if g['id'] in ini)
        base_avoid = sum(g['y'] * g['n'] for g in geo if g['id'] in ini) / base_n
        pooled = {}
        for key in rows[0]['init']:
            n = sum(r['init'][key]['n'] for r in rows)
            pooled[key] = dict(n=n, **{k: sum(r['init'][key][k] for r in rows) / n for k in ('intact', 'clear', 'valid', 'via_b')})
            pooled[key]['avoid'] = 1 - pooled[key]['via_b']
        d192, d64 = pooled['route_d_k192'], pooled['route_d_k64']
        if d192['avoid'] >= 0.5 and d192['avoid'] >= base_avoid + 0.3 and d192['intact'] >= 0.5:
            d4 = 'supports: the early coarse shape sets the corridor'
        elif d64['avoid'] < base_avoid + 0.2:
            d4 = 'does not support: the model returns to the shortest corridor even from an almost finished detour'
        else:
            d4 = 'uncertain'
        out['F4'] = dict(n_scen=len(rows), baseline_avoid=base_avoid, pooled=pooled, decision=d4)
        lines += ['## F4: denoising started from a noised hand-made route', '',
                  f'Baseline (E1/E1r, plain sampling): not through b {base_avoid:.2f} ({len(rows)} scenarios).', '',
                  '| start route | k (steps left) | not through b | intact | valid | n |', '|---|---|---|---|---|---|']
        for key, p in pooled.items():
            route, k = key.rsplit('_k', 1)
            lab = 'detour R_d' if route == 'route_d' else 'through b R_b'
            lines.append(f"| {lab} | {k} | {p['avoid']:.2f} | {p['intact']:.2f} | {p['valid']:.2f} | {p['n']} |")
        lines += ['', '| id | y | ' + ' | '.join(f"{'d' if k.startswith('route_d') else 'b'} k{k.rsplit('_k', 1)[1]} avoid/valid" for k in pooled) + ' |',
                  '|---|---|' + '---|' * len(pooled)]
        for r in rows:
            g = next(g for g in geo if g['id'] == r['id'])
            lines.append(f"| {r['id']} | {g['y']:.2f} | " + ' | '.join(
                f"{1 - r['init'][k]['via_b'] / r['init'][k]['n']:.2f} / {r['init'][k]['valid'] / r['init'][k]['n']:.2f}" for k in pooled) + ' |')
        lines += ['', f'**F4 decision: {d4}.**', '']

    # F5 (plan section 6)
    nz = {r['id']: r for r in (json.load(open(f)) for f in glob.glob(os.path.join(BASE, 'noise', '*.json')))}
    if nz:
        rows = [nz[i] for i in ids if i in nz]
        ws, qs = [], []
        lines += ['## F5: same starting noise x_T, different noise afterwards (plan section 6)', '',
                  '| id | via b | p | q (x_T irrelevant) | w (same x_T) | via b per x_T (of 8) |', '|---|---|---|---|---|---|']
        for r in rows:
            via = r['via_b']
            flat = [v for g in via for v in g]
            p = sum(flat) / len(flat)
            q = p * p + (1 - p) * (1 - p)
            pairs = [(g[i] == g[j]) for g in via for i in range(len(g)) for j in range(i + 1, len(g))]
            w = sum(pairs) / len(pairs)
            ws.append(w)
            qs.append(q)
            lines.append(f"| {r['id']} | {sum(flat)}/{len(flat)} | {p:.2f} | {q:.2f} | {w:.2f} | {' '.join(str(sum(g)) for g in via)} |")
        W, Q = sum(ws) / len(ws), sum(qs) / len(qs)
        d5 = 'supports: the starting noise decides the corridor' if W >= 0.9 and W - Q >= 0.15 else \
            'does not support: later noise matters as much as x_T' if W - Q <= 0.05 else 'uncertain (both matter)'
        out['F5'] = dict(n_scen=len(rows), W=W, Q=Q, decision=d5)
        lines += ['', f'W = {W:.2f}, Q = {Q:.2f}, W - Q = {W - Q:.2f}.', f'**F5 decision: {d5}.**', '']

    # F6 (plan section 7)
    f6 = os.path.join(BASE, 'dataset.json')
    if os.path.exists(f6):
        d6 = json.load(open(f6))
        rows = [r for r in d6['rows'] if r['n'] >= 10]
        excl = [r['id'] for r in d6['rows'] if r['n'] < 10]
        r6, p6 = perm_p([r['x6'] for r in rows], [r['y'] for r in rows])
        dec6 = 'supports time-matched data frequency' if r6 >= 0.5 and p6 < 0.05 else \
            'does not support time-matched data frequency' if r6 < 0.25 else 'uncertain'
        mad = sum(abs(r['x6'] - r['y']) for r in rows) / len(rows)
        via, tb = sum(r['via'] for r in rows), sum(r['turn_back1_via'] for r in rows)
        out['F6'] = dict(window=d6['window'], n_scen=len(rows), excluded=excl, rho=r6, p=p6, decision=dec6, mean_abs_diff=mad,
                         via_turn_back=[tb, via])
        lines += [f"## F6: training segments A -> G that take {d6['window'][0]}-{d6['window'][1]} steps (plan section 7)", '',
                  '| id | segments | avoiding b | x6 | y | via b with a turn-back (depth 1 / 2) |', '|---|---|---|---|---|---|']
        lines += [f"| {r['id']} | {r['n']} | {r['avoid']} | {fmt(r['x6'])} | {r['y']:.2f} | "
                  f"{r['turn_back1_via']} / {r['turn_back2_via']} of {r['via']} |" for r in d6['rows']]
        lines += ['', f'Excluded (< 10 segments): {excl or "none"}.',
                  f'Spearman rho(x6, y) = {r6:.2f}, permutation p = {p6:.3f} ({len(rows)} scenarios); mean |x6 - y| = {mad:.2f}.',
                  f'Segments through b with a turn-back: {tb}/{via}.', f'**F6 decision: {dec6}.**', '']
    # F7 (plan section 8)
    f7 = os.path.join(BASE, 'window.json')
    if os.path.exists(f7):
        d7 = json.load(open(f7))
        rows = [r for r in d7['rows'] if r['clusters'] >= 5]
        excl = [r['id'] for r in d7['rows'] if r['clusters'] < 5]
        r7, p7 = perm_p([r['x7'] for r in rows], [r['y'] for r in rows])
        dec7 = 'supports the training distribution under the planning endpoints' if r7 >= 0.5 and p7 < 0.05 else \
            'does not support the training distribution under the planning endpoints' if r7 < 0.25 else 'uncertain'
        mad = sum(abs(r['x7'] - r['y']) for r in rows) / len(rows)
        mx, my = st.mean(r['x7'] for r in rows), st.mean(r['y'] for r in rows)
        rc, pc = perm_p([r['x7_cluster'] for r in rows], [r['y'] for r in rows])
        out['F7'] = dict(n_scen=len(rows), excluded=excl, rho=r7, p=p7, decision=dec7, mean_abs_diff=mad, mean_x7=mx, mean_y=my,
                         rho_cluster=rc, p_cluster=pc)
        lines += ['## F7: training windows with step 0 in A and step 383 in G (plan section 8)', '',
                  '| id | windows | clusters | avoiding b | x7 | x7 by cluster | y |', '|---|---|---|---|---|---|---|']
        lines += [f"| {r['id']} | {r['n']} | {r['clusters']} | {r['avoid']} | {fmt(r['x7'])} | {fmt(r['x7_cluster'])} | "
                  f"{r['y']:.2f} |" for r in d7['rows']]
        lines += ['', f'Excluded (< 5 clusters): {excl or "none"}.',
                  f'Spearman rho(x7, y) = {r7:.2f}, permutation p = {p7:.3f} ({len(rows)} scenarios); mean |x7 - y| = {mad:.2f}; '
                  f'mean x7 {mx:.2f} vs mean y {my:.2f}.',
                  f'By cluster (descriptive): rho = {rc:.2f}, p = {pc:.3f}.', f'**F7 decision: {dec7}.**', '']
    tbf = os.path.join(BASE, 'turnback.json')
    if os.path.exists(tbf):
        tbr = json.load(open(tbf))
        nv = sum(r['n_via'] for r in tbr)
        out['turnback'] = dict(n_via=nv, turn_back=sum(r['via_turn_back'] for r in tbr),
                               turn_back2=sum(r['via_turn_back2'] for r in tbr), along_detour=sum(r['via_along_detour'] for r in tbr))
        lines += ['## Plans through b: how they spend the spare time (descriptive, plan section 7)', '',
                  '| id | through b | with a turn-back | depth >= 2 | median depth | start along the detour | not through b |',
                  '|---|---|---|---|---|---|---|']
        lines += [f"| {r['id']} | {r['n_via']} | {r['via_turn_back']} | {r['via_turn_back2']} | {r['via_depth_median']} | "
                  f"{r['via_along_detour']} | {r['n_avoid']} |" for r in tbr]
        t = out['turnback']
        lines += ['', f"Pooled: {t['turn_back']}/{nv} plans through b have a turn-back ({t['turn_back2']} of depth >= 2); "
                  f"{t['along_detour']} first step onto the detour.", '']

    with open(os.path.join(BASE, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(BASE, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
