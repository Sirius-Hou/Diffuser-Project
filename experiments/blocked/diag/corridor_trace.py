"""
F3 and F4 of experiment_progress/走廊偏好實驗計畫_2026-11-27.md (exploratory, rules fixed in the plan).
The frozen Diffuser plans alone (it is never told about the obstacle); b is only used to score the plans.

  --mode trace : F3. Re-run the E1 / E1r sampling of the 24 S scenarios with the same seeds, and record the model's
                 prediction of the clean trajectory (x0-hat) at each of the 256 denoising steps through a forward hook
                 (sampling itself is unchanged). Step j = 0 is the first, noisiest step.
  --mode noise : F5 (added 11/27 after F3, before running it; plan section 6). For the scenarios with mixed outcomes,
                 16 starting noises x_T, each denoised 8 times with different noise in the later steps.
  --mode init  : F4. Start denoising from a noised hand-made trajectory along the detour R_d or the route through b
                 R_b (constant speed, at G on step 359), with k in {64, 128, 192} of the 256 steps left.

    bash experiments/common/setup/run.sh python experiments/blocked/diag/corridor_trace.py --mode trace --ids e1_00 e1_01
One JSON per scenario in results/diag/corridor/<mode>/, so several processes can share the output directory.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from diffuser.models.helpers import apply_conditioning  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds  # noqa: E402
from budget_sampling import SEED_BASE, score  # noqa: E402
from corridor_geometry import seg_dist  # noqa: E402
from denoise_probe import polyline, reference  # noqa: E402

H = 384
BATCH = 16
AGREE_AT = (0, 16, 32, 64, 128)
SAVE_AT = (0, 16, 32, 64, 128, 255)
INIT_K = (64, 128, 192)
INIT_N = 32
INIT_SEED = 850000
NOISE_SEED = 870000
N_XT, N_LATER = 16, 8
BUDGET = os.path.join(REPO, 'experiments/blocked/results/diag/budget')
OUT = os.path.join(REPO, 'experiments/blocked/results/diag/corridor')


def scenarios():
    """The 24 S scenarios with the index that set their E1 / E1r seed, and the F1 routes."""
    geo = {g['id']: g for g in json.load(open(os.path.join(OUT, 'geometry.json')))}
    out = []
    for exp in ('e1', 'e1r'):
        for i, s in enumerate(json.load(open(os.path.join(BUDGET, exp, 'scenarios.json')))['scenarios']):
            if s['id'] in geo:
                out.append(dict(s, exp=exp, index=i, route_b=geo[s['id']]['route_b'], route_d=geo[s['id']]['route_d']))
    return out


def poly_dist(p, cells):
    """Mean over points p [..., N, 2] of the distance to the polyline through the cell centres."""
    c = np.array([mu.cell_center(tuple(x)) for x in cells], dtype=np.float64)
    a, v = c[:-1], c[1:] - c[:-1]                                     # segments [M, 2]
    u = np.clip(np.einsum('...nmk,mk->...nm', p[..., :, None, :] - a, v) / (v * v).sum(1), 0.0, 1.0)
    d = np.linalg.norm(p[..., :, None, :] - (a + u[..., None] * v), axis=-1)
    return d.min(-1).mean(-1)


def in_cell(xy, b):
    """[..., N, 2] -> [...] whether any point lies in cell b (same rule as mu.xy_to_cell)."""
    c = np.floor(xy + mu.CELL_OFFSET + 0.5).astype(int)
    return ((c[..., 0] == b[0]) & (c[..., 1] == b[1])).any(-1)


def run_trace(planner, maze, s, save_dir):
    norm, A = planner.policy.normalizer, planner.diffusion.action_dim
    steps = []
    hook = planner.diffusion.model.register_forward_hook(
        lambda mod, inp, out: steps.append(norm.unnormalize(out.detach().clamp(-1, 1)[..., A:].numpy(), 'observations')[..., :2]))
    state, target, b = np.r_[mu.cell_center(tuple(s['A'])), 0.0, 0.0], mu.cell_center(tuple(s['G'])), tuple(s['b'])
    obstacle = ob.Obstacle.full(b)
    set_all_seeds(SEED_BASE[s['exp']] + 1000 * s['index'] + H)          # as budget_sampling.py
    xs, finals = [], []
    for _ in range(0, 64, BATCH):
        steps.clear()
        plans, _, _ = planner.plan(state, target, n=BATCH, horizon=H)
        assert len(steps) == planner.n_diffusion_steps
        xs.append(np.stack(steps, 1).astype(np.float32))                  # [batch, 256, H, 2]
        finals += [score(maze, p, obstacle, b, 0.05) for p in plans]
    hook.remove()
    full = np.concatenate(xs)                                             # [64, 256, H, 2]
    x = full[:, :, 1:H - 1]          # x0-hat before conditioning: its first and last points are not the plan's
    v = in_cell(x, b)                                                     # [64, 256]
    fv = np.array([f['via_b'] for f in finals])
    stored = json.load(open(os.path.join(BUDGET, s['exp'], 'rows', f"{s['id']}.json")))['samples']
    same = [f['via_b'] for f in finals] == [r['via_b'] for r in stored] and [f['valid'] for f in finals] == [r['valid'] for r in stored]
    lock = np.array([int(np.max(np.nonzero(row != f)[0]) + 1) if (row != f).any() else 0 for row, f in zip(v, fv)])
    a, g = mu.cell_center(tuple(s['A'])), mu.cell_center(tuple(s['G']))
    d_line = seg_dist(x.reshape(-1, 2).astype(np.float64), a, g).reshape(x.shape[:3]).mean(-1)
    d_b = np.stack([poly_dist(xi.astype(np.float64), s['route_b']) for xi in x])      # per sample: keeps memory small
    d_d = np.stack([poly_dist(xi.astype(np.float64), s['route_d']) for xi in x])
    spread = [float(np.linalg.norm(x[:, j, None] - x[None, :, j], axis=-1).mean(-1)[np.triu_indices(len(x), 1)].mean())
              for j in range(x.shape[1])]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, f"{s['id']}.npz"), steps=np.array(SAVE_AT),
                            xy=full[:, list(SAVE_AT)].astype(np.float16), final_via_b=fv)
    return dict(reproduced=bool(same), n=len(finals), final_via_b=fv.tolist(), final_valid=[f['valid'] for f in finals],
                lock=lock.tolist(), via_b_share=v.mean(0).tolist(), agree={j: float((v[:, j] == fv).mean()) for j in AGREE_AT},
                d_line=d_line.mean(0).tolist(), d_b=d_b.mean(0).tolist(), d_d=d_d.mean(0).tolist(), spread=spread)


def run_init(planner, maze, env, s, gidx):
    dif, policy = planner.diffusion, planner.policy
    norm, A = policy.normalizer, dif.action_dim
    state, target, b = np.r_[mu.cell_center(tuple(s['A'])), 0.0, 0.0], mu.cell_center(tuple(s['G'])), tuple(s['b'])
    obstacle = ob.Obstacle.full(b)
    cond = policy._format_conditions({0: state, H - 1: np.array([target[0], target[1], 0.0, 0.0])}, INIT_N)
    out = {}
    for r_i, route in enumerate(('route_d', 'route_b')):
        pts, cum = polyline([tuple(c) for c in s[route]])
        ref = reference(pts, cum)
        obs = np.c_[ref, np.r_[np.diff(ref, axis=0), np.zeros((1, 2))] / env.dt]
        x0 = np.concatenate([norm.normalize(np.zeros((H, A)), 'actions'), norm.normalize(obs, 'observations')], -1)
        x0 = torch.tensor(x0, dtype=torch.float32)[None].repeat(INIT_N, 1, 1)
        for k_i, k in enumerate(INIT_K):
            set_all_seeds(INIT_SEED + 1000 * gidx + 10 * k_i + r_i)
            with torch.no_grad():
                x = apply_conditioning(dif.q_sample(x0, torch.full((INIT_N,), k - 1, dtype=torch.long)), cond, A)
                for i in reversed(range(k)):
                    x = apply_conditioning(dif.p_sample(x, cond, torch.full((INIT_N,), i, dtype=torch.long)), cond, A)
            plans = norm.unnormalize(x[..., A:].numpy(), 'observations')
            sc = [score(maze, p, obstacle, b, 0.05) for p in plans]
            out[f'{route}_k{k}'] = {key: sum(x_[key] for x_ in sc) for key in ('intact', 'clear', 'valid', 'via_b')}
            out[f'{route}_k{k}']['n'] = len(sc)
            print(f"[F4] {s['id']:7s} {route} k {k:3d} | " + ' '.join(f'{key} {v:2d}' for key, v in out[f'{route}_k{k}'].items()),
                  flush=True)
    return dict(init=out)


def run_noise(planner, maze, s, gidx):
    dif, policy = planner.diffusion, planner.policy
    norm, A = policy.normalizer, dif.action_dim
    state, target, b = np.r_[mu.cell_center(tuple(s['A'])), 0.0, 0.0], mu.cell_center(tuple(s['G'])), tuple(s['b'])
    obstacle = ob.Obstacle.full(b)
    n, chunk = N_XT * N_LATER, 4                                                # 4 x_T (32 samples) per batch
    cond = policy._format_conditions({0: state, H - 1: np.array([target[0], target[1], 0.0, 0.0])}, chunk * N_LATER)
    set_all_seeds(NOISE_SEED + 1000 * gidx)
    x_t = torch.randn((N_XT, H, dif.transition_dim))
    plans = []
    for c0 in range(0, N_XT, chunk):
        group = torch.tensor(np.repeat(np.arange(c0, c0 + chunk), N_LATER))
        with torch.no_grad():
            x = apply_conditioning(x_t[group].clone(), cond, A)                 # each x_T repeated N_LATER times
            for i in reversed(range(dif.n_timesteps)):                          # later noise: torch.randn_like, all distinct
                x = apply_conditioning(dif.p_sample(x, cond, torch.full((len(group),), i, dtype=torch.long)), cond, A)
        plans.append(norm.unnormalize(x[..., A:].numpy(), 'observations'))
    plans = np.concatenate(plans)
    sc = [score(maze, p, obstacle, b, 0.05) for p in plans]
    via = np.array([x_['via_b'] for x_ in sc]).reshape(N_XT, N_LATER)
    print(f"[F5] {s['id']:7s} via_b {int(via.sum())}/{n} | per x_T " + ' '.join(str(int(v)) for v in via.sum(1)), flush=True)
    return dict(n_xt=N_XT, n_later=N_LATER, via_b=via.astype(int).tolist(),
                valid=np.array([x_['valid'] for x_ in sc]).reshape(N_XT, N_LATER).astype(int).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['trace', 'init', 'noise'], required=True)
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--save_dir', default=None, help='trace: also save x0-hat at a few steps (.npz)')
    args = ap.parse_args()
    out = os.path.join(OUT, args.mode)
    os.makedirs(out, exist_ok=True)
    env = load_env('maze2d-large-v1')
    maze = env.maze_arr
    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu', threads=args.threads)
    for gidx, s in enumerate(scenarios()):
        dst = os.path.join(out, f"{s['id']}.json")
        if (args.ids is not None and s['id'] not in args.ids) or os.path.exists(dst):
            continue
        res = run_trace(planner, maze, s, args.save_dir) if args.mode == 'trace' else \
            run_init(planner, maze, env, s, gidx) if args.mode == 'init' else run_noise(planner, maze, s, gidx)
        row = dict(id=s['id'], exp=s['exp'], index=s['index'], group=s['group'], A=s['A'], G=s['G'], b=s['b'],
                   base=s['base'], total=s['total'], extra=s['extra'], **res)
        with open(dst, 'w') as f:
            json.dump(row, f)
        if args.mode == 'trace':
            print(f"[F3] {s['id']:7s} reproduced {res['reproduced']} | via_b {sum(res['final_via_b'])}/64 | "
                  f"lock median {np.median(res['lock']):.0f} | agree " +
                  ' '.join(f'{j}:{a:.2f}' for j, a in res['agree'].items()), flush=True)


if __name__ == '__main__':
    main()
