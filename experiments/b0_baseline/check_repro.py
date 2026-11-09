"""
Reproducibility check: compare per-episode outputs of two runs of the same seeds.

    python experiments/b0_baseline/check_repro.py RUN_DIR_A RUN_DIR_B [--tag_b _rep] [--out FILE]

Plans and rollouts must be bit-identical, and every metric except wall-clock time must match.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

TIMING = {'planning_time_s', 'episode_time_s'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--tag_a', default='')
    ap.add_argument('--tag_b', default='')
    ap.add_argument('--out')
    args = ap.parse_args()

    files_a = sorted(f for f in glob.glob(os.path.join(args.a, 'episodes', f'seed[0-9][0-9][0-9]{args.tag_a}.json')))
    checks, all_ok = [], True
    for fa in files_a:
        seed = int(os.path.basename(fa)[4:7])
        fb = os.path.join(args.b, 'episodes', f'seed{seed:03d}{args.tag_b}.json')
        if not os.path.exists(fb):
            continue
        ja, jb = json.load(open(fa)), json.load(open(fb))
        na, nb = np.load(fa[:-5] + '.npz'), np.load(fb[:-5] + '.npz')
        c = dict(seed=seed,
                 plan_identical=bool(np.array_equal(na['plan'], nb['plan'])),
                 rollout_identical=bool(np.array_equal(na['rollout_xy'], nb['rollout_xy'])),
                 max_abs_plan_diff=float(np.abs(na['plan'] - nb['plan']).max()),
                 differing_metrics=[k for k in ja if k not in TIMING and ja.get(k) != jb.get(k)],
                 planning_time_s=[ja['planning_time_s'], jb['planning_time_s']])
        c['ok'] = c['plan_identical'] and c['rollout_identical'] and not c['differing_metrics']
        all_ok &= c['ok']
        checks.append(c)
        print(f"seed {seed:3d}: plan identical={c['plan_identical']} rollout identical={c['rollout_identical']} "
              f"differing metrics={c['differing_metrics']}")
    res = dict(run_a=args.a, run_b=args.b, tag_a=args.tag_a, tag_b=args.tag_b,
               n_compared=len(checks), all_identical=bool(all_ok and checks), checks=checks)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
    print('REPRODUCIBLE' if res['all_identical'] else 'NOT REPRODUCIBLE / nothing compared')
    sys.exit(0 if res['all_identical'] else 1)


if __name__ == '__main__':
    main()
