"""
Summary of interior_sampling.py (experiment 1 of final_project/補強實驗計畫_2026-11-23.md) next to G0 (no via
points, guided_sampling summary), with the pre-registered decision.

    python experiments/blocked/diag/summarize_interior.py       (plain Python)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results', 'diag')
GROUPS = [('I1_junctions_h384', 'junctions, H 384'), ('I2_corners_h384', 'junctions + corners, H 384'),
          ('I3_corners_matched', 'junctions + corners, H by length'),
          ('I4_corners_matched_ws5', 'as I3 + guidance (obstacle, walls, smooth)')]


def main():
    g0 = json.load(open(os.path.join(RES, 'guided_sampling', 'summary.json')))
    lines = ['# Interior inpainting (proposal option b), offline', '',
             'Valid plans that also visit a detour-only cell, out of 64 per scenario. Decision rule (written before '
             'running): a group passes if >= 2 of the 13 scenarios with a detour >= 8 cells longer have >= 4/64.', '',
             '| Group | Via points | Detour >= 8 longer | Detour <= 6 longer | Scenarios >= 4/64 (>= 8) | Failures (>= 8) | '
             'Median max jump (>= 8) | Via points passed (>= 8, mean) |', '|---|---|---|---|---|---|---|---|']
    out = {}
    for tag, desc in GROUPS:
        p = os.path.join(RES, 'interior_sampling', tag, 'rows.json')
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))['rows']
        long_ = [r for r in rows if r['detour_cells'] - r['base_cells'] >= 8]
        short = [r for r in rows if r['detour_cells'] - r['base_cells'] < 8]
        k_long = sum(r['n_valid_via_detour'] for r in long_)
        k_short = sum(r['n_valid_via_detour'] for r in short)
        n_pass = sum(r['n_valid_via_detour'] >= 4 for r in long_)
        fail = {k: sum(r['fail'][k] for r in long_) for k in ('still_hit', 'split', 'wall')}
        jumps = sorted(r['median_max_jump'] for r in long_)
        via = sum(r['mean_via_passed'] / max(r['n_via'], 1) for r in long_) / max(len(long_), 1)
        out[tag] = dict(n_long=len(long_), n_short=len(short), valid_detour_long=k_long, valid_detour_short=k_short,
                        scenarios_pass=n_pass, passes=n_pass >= 2, fail_long=fail,
                        median_max_jump_long=jumps[len(jumps) // 2] if jumps else None, via_passed_frac_long=via)
        lines.append(f'| {tag} | {desc} | {k_long}/{64 * len(long_)} | ' +
                     (f'{k_short}/{64 * len(short)}' if short else 'not run') +
                     f' | {n_pass}/{len(long_)} | {fail} | {out[tag]["median_max_jump_long"]:.2f} | {100 * via:.0f}% |')
    g = g0['full_G0']
    lines += ['', f"G0 (no via points, same states and seeds; guided_sampling): detour >= 8 longer "
                  f"{g['long']['valid_via']}/{g['long']['n']}, <= 6 longer {g['short']['valid_via']}/{g['short']['n']}.", '',
              f"Decision: {'passes (' + ', '.join(t for t, v in out.items() if v['passes']) + ')' if any(v['passes'] for v in out.values()) else 'no group passes: interior inpainting does not move the plan onto the detour'}."]
    with open(os.path.join(RES, 'interior_sampling', 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES, 'interior_sampling', 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
