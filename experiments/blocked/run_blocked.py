"""
Run one or more methods on the frozen scenario list. One JSON (+ npz, png) per episode.

    python experiments/blocked/run_blocked.py --config experiments/blocked/configs/smoke.json
    python experiments/blocked/run_blocked.py --config experiments/blocked/configs/main.json --methods B2 Ours
    python experiments/blocked/run_blocked.py --config ... --ids full_s004 full_s006   # chosen scenarios
    python experiments/blocked/run_blocked.py --config ... --fake                       # logic test, no mujoco

Episodes already on disk are skipped (so a Colab disconnect only loses the current episode);
pass --overwrite to redo them.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.blocked.runtime import REPO, build, git_commit  # noqa: E402
from experiments.common.methods import make_method  # noqa: E402
from experiments.common.rollout import run_episode, render_episode  # noqa: E402


def load_scenarios(cfg, args):
    path = args.scenarios or cfg['scenarios']
    data = json.load(open(os.path.join(REPO, path)))
    scen = data['scenarios']
    kinds = cfg.get('kinds')
    if kinds:
        scen = [s for s in scen if s['obstacles'][0]['kind'] in kinds]
    if args.ids:
        scen = [s for s in scen if s['id'] in args.ids]
    if cfg.get('limit'):
        scen = scen[:cfg['limit']]
    if args.limit:
        scen = scen[:args.limit]
    return scen, path, data['meta']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--scenarios')
    ap.add_argument('--out')
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no_png', action='store_true')
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    methods = args.methods or cfg['methods']
    scen, scen_path, scen_meta = load_scenarios(cfg, args)
    out = os.path.join(REPO, args.out or cfg['out'])
    env, planner, maze, contact = build(cfg, args.fake)
    if scen_meta.get('fake') and not (args.fake or cfg.get('fake')):
        print('[run] WARNING: this scenario file was made with the FAKE planner; '
              'regenerate it with the real model before trusting results')
    save_png = cfg.get('save_png', True) and not args.no_png
    if save_png:
        import matplotlib
        matplotlib.use('Agg')

    for name in methods:
        mdir = os.path.join(out, name, 'episodes')
        os.makedirs(mdir, exist_ok=True)
        meta = dict(config=cfg, config_path=os.path.relpath(os.path.abspath(args.config), REPO),
                    method=name, scenarios=scen_path, scenario_ids=[s['id'] for s in scen],
                    git_commit=git_commit(), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                    python=sys.version.split()[0], fake=bool(args.fake or cfg.get('fake')),
                    **(planner.meta() if hasattr(planner, 'meta') else {}))
        with open(os.path.join(out, name, 'run_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        t0 = time.perf_counter()
        for s in scen:
            stem = os.path.join(mdir, s['id'])
            if os.path.exists(stem + '.json') and not args.overwrite:
                continue
            method = make_method(name, cfg)
            rec, traj, ctx = run_episode(env, planner, method, s, cfg, maze, contact)
            rec['events'] = ctx.events
            rec['plan_calls'] = ctx.plan_calls
            with open(stem + '.json', 'w') as f:
                json.dump(rec, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
            np.savez_compressed(stem + '.npz', **traj)
            if save_png:
                render_episode(stem + '.png', maze, rec, traj, ctx, env._target, cfg.get('success_radius', 0.5))
            print(f"[{name}] {s['id']:16s} success={rec['success']!s:5} {rec['failure_reason'] or '':20s} "
                  f"steps={rec['steps']:3d} replans={rec['n_replans']} plans={rec['n_plans']:3d} "
                  f"astar={rec['n_astar']} plan_time={rec['planning_time_s']:.1f}s", flush=True)
        print(f'[{name}] done in {time.perf_counter() - t0:.1f}s -> {os.path.relpath(os.path.join(out, name), REPO)}',
              flush=True)


if __name__ == '__main__':
    main()
