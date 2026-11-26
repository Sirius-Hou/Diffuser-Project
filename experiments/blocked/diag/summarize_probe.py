"""
Summary and decisions for denoise_probe.py, with the rules of experiment_progress/去噪探針實驗計畫_2026-11-26.md
(section 4). Physical trajectories that miss G by more than 0.5 or have points in a wall are excluded (section 2).

    python experiments/blocked/diag/summarize_probe.py        (plain Python)
Writes results/diag/denoise_probe/summary.{md,json}.
"""
import json
import os
import statistics as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BASE = os.path.join(REPO, 'experiments/blocked/results/diag/denoise_probe')


def pctl(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def main():
    d = json.load(open(os.path.join(BASE, 'scores.json')))
    items = d['items']
    data = [r['score'] for r in items if r['kind'] == 'DATA']
    p90 = pctl(data, 0.9)
    ok = lambda r: r['kind'] in ('KROUTE', 'WALL', 'DATA') or (r['final_dist'] <= 0.5 and r['in_wall'] == 0)
    excluded = [(r['kind'], r['key'], r['A'], r['G']) for r in items if not ok(r)]
    use = [r for r in items if ok(r)]
    sel = lambda kind, keys: [r for r in use if r['kind'] == kind and r['key'] in keys]
    high = lambda rs: sum(r['score'] > p90 for r in rs)

    mid = sel('SHORT', (11, 13, 15))
    c1 = high(mid) <= len(mid) / 3
    wall, kroute = sel('WALL', (11, 13, 15)), sel('KROUTE', (11, 13, 15))
    c2 = high(wall) >= 0.8 * len(wall) and high(kroute) <= len(kroute) / 3
    fast = sel('SHORT', (17, 19, 21))
    if not (c1 and c2):
        d_len = d_shape = 'probe not usable (calibration failed)'
    else:
        if high(fast) >= 0.75 * len(fast) and high(mid) <= len(mid) / 3:
            d_len = 'supports the time budget'
        elif high(fast) <= len(fast) / 3:
            d_len = 'does not support the time budget'
        else:
            d_len = 'uncertain'
        d_shape = None
    pairs = []
    for T in (9, 11, 13, 15):
        for a in sel('DETOUR', (T,)):
            for b in sel('SHORT', (T,)):
                pairs.append(a['score'] > b['score'])
    A = sum(pairs) / len(pairs) if pairs else float('nan')
    if d_shape is None:
        d_shape = 'supports route rarity' if A >= 0.75 else 'does not support route rarity' if A <= 0.60 else 'uncertain'

    lines = ['# Denoising probe: summary (exploratory)', '',
             'Rules: experiment_progress/去噪探針實驗計畫_2026-11-26.md, section 4. "High" = above the 90th percentile '
             f'of the training windows (DATA p90 = {p90:.4f}).', '',
             f'Excluded physical trajectories (missed G by > 0.5 or points in a wall): {len(excluded)} {excluded or ""}', '',
             '| set | key | n | median score | high (> DATA p90) | median arc | median speed per step |',
             '|---|---|---|---|---|---|---|']
    for kind in ('DATA', 'SHORT', 'DETOUR', 'KROUTE', 'WALL'):
        keys = sorted({r['key'] for r in use if r['kind'] == kind}) if kind != 'DATA' else ['all']
        for k in keys:
            rs = [r for r in use if r['kind'] == kind and (kind == 'DATA' or r['key'] == k)]
            lines.append(f"| {kind} | {k} | {len(rs)} | {st.median(r['score'] for r in rs):.4f} | {high(rs)}/{len(rs)} | "
                         f"{st.median(r['arc'] for r in rs):.1f} | {st.median(r['speed_median'] for r in rs):.4f} |")
    lines += ['', f'C1 (SHORT 11-15 at most 1/3 high): {high(mid)}/{len(mid)} high -> {"pass" if c1 else "FAIL"}',
              f'C2 (WALL >= 80% high and KROUTE at most 1/3 high): WALL {high(wall)}/{len(wall)}, KROUTE '
              f'{high(kroute)}/{len(kroute)} -> {"pass" if c2 else "FAIL"}', '',
              f'Length / speed: SHORT 17-21 high {high(fast)}/{len(fast)}, SHORT 11-15 high {high(mid)}/{len(mid)}',
              f'**Length / speed decision: {d_len}.**', '',
              f'Shape: share of same-length (detour, shortest) pairs, T in 9-15, where the detour scores higher: '
              f'A = {A:.2f} ({sum(pairs)}/{len(pairs)})', f'**Shape decision: {d_shape}.**', '',
              '## Per noise level (median score)', '', '| set | ' + ' | '.join(f't={t}' for t in d['meta']['t_levels']) + ' |',
              '|---|' + '---|' * len(d['meta']['t_levels'])]
    for kind in ('DATA', 'SHORT', 'DETOUR', 'KROUTE', 'WALL'):
        rs = [r for r in use if r['kind'] == kind]
        lines.append(f'| {kind} | ' + ' | '.join(f"{st.median(r['per_t'][str(t)]['obs'] for r in rs):.4f}"
                                                for t in d['meta']['t_levels']) + ' |')
    out = dict(data_p90=p90, excluded=excluded, c1=c1, c2=c2, length_decision=d_len, shape_decision=d_shape, shape_A=A,
               n_pairs=len(pairs), fast_high=[high(fast), len(fast)], mid_high=[high(mid), len(mid)],
               wall_high=[high(wall), len(wall)], kroute_high=[high(kroute), len(kroute)])
    with open(os.path.join(BASE, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(BASE, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
