"""
Denoising probe (experiment_progress/去噪探針實驗計畫_2026-11-26.md; exploratory, decision rules fixed in the plan).

Instead of sampling, hand the frozen Diffuser a finished trajectory, noise it, and measure how well the model
denoises it: the training loss of that trajectory, a rough proxy for -log p(tau). Every probe trajectory goes from
A to G in exactly 384 steps (arriving at step 359, then held at G), so a longer route means a faster ball.

  DATA      : random 384-step windows of the training set
  SHORT(L)  : shortest routes of L cells, ball driven by a PD controller along the cell centres (MuJoCo)
  DETOUR(T) : routes of T cells that avoid a blocked cell b on the shortest route (extra >= 4 cells)
  KROUTE    : kinematic copies of SHORT(11, 13, 15) (positions interpolated, no physics)
  WALL      : kinematic straight lines from A to G, through the walls

Score: mean squared denoising error of the normalised observations (steps 1..382), conditioned like planning
(start observation at step 0, (G, 0, 0) at step 383), averaged over t in T_LEVELS and K shared noise draws.

    bash experiments/common/setup/run.sh python experiments/blocked/diag/denoise_probe.py
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffuser.models.helpers import apply_conditioning  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds  # noqa: E402
from budget_sampling import e1_candidates, free_cells  # noqa: E402

H, ARRIVE = 384, 359
T_LEVELS = (8, 32, 64, 128, 192)
K = 8
SHORT_L = (5, 7, 9, 11, 13, 15, 17, 19, 21)
DETOUR_T = (9, 11, 13, 15, 17, 19, 21)
PER_SET = 6
PICK_SEED = 20261126


def polyline(cells):
    pts = np.array([mu.cell_center(c) for c in cells], dtype=np.float64)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts, cum


def point_at(pts, cum, s):
    if s >= cum[-1]:
        return pts[-1].copy()
    i = int(np.searchsorted(cum, s, side='right')) - 1
    f = (s - cum[i]) / (cum[i + 1] - cum[i])
    return pts[i] + f * (pts[i + 1] - pts[i])


def reference(pts, cum):
    """Reference position at every step: constant speed along the line, at the end on step ARRIVE."""
    return np.array([point_at(pts, cum, min(t / ARRIVE, 1.0) * cum[-1]) for t in range(H)])


def rollout(env, ref, kp, kv):
    """Drive the ball along `ref` with velocity feed-forward plus PD; returns observations [H, 4], actions [H, 2]."""
    env.reset()
    env.set_state(ref[0].copy(), np.zeros(2))
    vel_ref = np.r_[np.diff(ref, axis=0), np.zeros((1, 2))] / env.dt
    obs, act = [], []
    for t in range(H):
        s = env.state_vector().copy()
        v_des = vel_ref[t] + kp * (ref[t] - s[:2])
        a = np.clip(kv * (v_des - s[2:]), -1.0, 1.0)
        obs.append(s)
        act.append(a)
        env.step(a)
    return np.array(obs), np.array(act)


def kinematic(ref, dt, k2):
    """Positions from `ref`, velocity = displacement / dt, action = velocity change / k2 (fitted from rollouts)."""
    vel = np.r_[np.diff(ref, axis=0), np.zeros((1, 2))] / dt
    act = np.clip(np.r_[np.diff(vel, axis=0), np.zeros((1, 2))] / k2, -1.0, 1.0)
    return np.c_[ref, vel], act


def pick_short(maze, rng):
    free = free_cells(maze)
    dist = {(a, g): len(mu.bfs_path(maze, a, g)) - 1 for a in free for g in free if a != g}
    return {L: rng.sample(sorted(p for p, d in dist.items() if d == L), min(PER_SET, sum(d == L for d in dist.values())))
            for L in SHORT_L}


def pick_detour(maze, rng):
    cands = [c for c in e1_candidates(maze) if c['extra'] >= 4]
    out = {}
    for T in DETOUR_T:
        mine = [c for c in cands if c['total'] == T]
        pairs = sorted({(tuple(c['A']), tuple(c['G'])) for c in mine})
        out[T] = []
        for a, g in rng.sample(pairs, min(PER_SET, len(pairs))):
            out[T].append(rng.choice(sorted((c for c in mine if (tuple(c['A']), tuple(c['G'])) == (a, g)), key=lambda c: c['b'])))
    return out


class Scorer:
    def __init__(self, planner):
        self.dif, self.policy = planner.diffusion, planner.policy
        self.norm, self.A = self.policy.normalizer, self.dif.action_dim
        g = torch.Generator().manual_seed(0)
        self.noise = torch.randn((K, H, self.dif.transition_dim), generator=g)

    @torch.no_grad()
    def __call__(self, obs, act, goal_xy):
        x0 = np.concatenate([self.norm.normalize(act, 'actions'), self.norm.normalize(obs, 'observations')], -1)
        x0 = torch.tensor(x0, dtype=torch.float32)[None].repeat(K, 1, 1)
        cond = self.policy._format_conditions({0: obs[0], H - 1: np.array([goal_xy[0], goal_xy[1], 0.0, 0.0])}, K)
        per_t = {}
        for t in T_LEVELS:
            tt = torch.full((K,), t, dtype=torch.long)
            xn = apply_conditioning(self.dif.q_sample(x0, tt, self.noise), cond, self.A)
            xr = apply_conditioning(self.dif.model(xn, cond, tt), cond, self.A)
            err = (xr - x0) ** 2
            per_t[t] = dict(obs=float(err[:, 1:H - 1, self.A:].mean()), act=float(err[:, 1:H - 1, :self.A].mean()))
        return dict(score=float(np.mean([v['obs'] for v in per_t.values()])), per_t=per_t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_data', type=int, default=200)
    ap.add_argument('--kp', type=float, default=5.0)
    ap.add_argument('--kv', type=float, default=1.0)
    ap.add_argument('--smoke', action='store_true', help='one trajectory per set, nothing written')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='experiments/blocked/results/diag/denoise_probe')
    args = ap.parse_args()
    out = os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)
    set_all_seeds(0)
    planner = FrozenDiffuserPlanner('maze2d-large-v1', 'diffusion/H384_T256', 'latest', 'cpu', threads=args.threads)
    env = load_env('maze2d-large-v1')
    maze = env.maze_arr
    score = Scorer(planner)
    rng = random.Random(PICK_SEED)
    short, detour = pick_short(maze, rng), pick_detour(maze, rng)
    if args.smoke:
        short = {L: v[:1] for L, v in short.items() if L in (5, 13, 21)}
        detour = {T: v[:1] for T, v in detour.items() if T in (13, 21)}
        args.n_data = 2
    items, t0 = [], time.perf_counter()

    def add(kind, key, obs, act, goal, **meta):
        xy = obs[:, :2]
        r = dict(kind=kind, key=key, **meta, final_dist=float(np.linalg.norm(xy[-1] - goal)),
                 in_wall=mu.count_points_in_wall(maze, xy), arc=mu.path_length(xy),
                 speed_median=float(np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1))), **score(obs, act, goal))
        items.append(r)
        print(f"[probe] {kind:7s} {str(key):4s} {meta.get('A', '')} -> {meta.get('G', '')} | arc {r['arc']:.1f} | "
              f"final {r['final_dist']:.2f} | in-wall {r['in_wall']} | score {r['score']:.4f} | {time.perf_counter() - t0:.0f}s", flush=True)

    # physical trajectories
    phys = []
    for L, pairs in short.items():
        for a, g in pairs:
            pts, cum = polyline(mu.bfs_path(maze, a, g))
            obs, act = rollout(env, reference(pts, cum), args.kp, args.kv)
            phys.append((obs, act))
            add('SHORT', L, obs, act, pts[-1], A=list(a), G=list(g), route_cells=L)
    for T, cs in detour.items():
        for c in cs:
            pts, cum = polyline(mu.bfs_path(maze, tuple(c['A']), tuple(c['G']), [tuple(c['b'])]))
            obs, act = rollout(env, reference(pts, cum), args.kp, args.kv)
            phys.append((obs, act))
            add('DETOUR', T, obs, act, pts[-1], A=c['A'], G=c['G'], b=c['b'], route_cells=T, extra=c['extra'])
    # fit the kinematic action scale from the rollouts: delta-velocity = k2 * action
    dv = np.concatenate([np.diff(o[:, 2:], axis=0) for o, _ in phys]).ravel()
    aa = np.concatenate([a[:-1] for _, a in phys]).ravel()
    k2 = float(dv @ aa / (aa @ aa))
    for L in (11, 13, 15):
        for a, g in short.get(L, []):
            pts, cum = polyline(mu.bfs_path(maze, a, g))
            obs, act = kinematic(reference(pts, cum), env.dt, k2)
            add('KROUTE', L, obs, act, pts[-1], A=list(a), G=list(g), route_cells=L)
            line = np.array([mu.cell_center(a), mu.cell_center(g)], dtype=np.float64)
            obs, act = kinematic(reference(line, np.r_[0.0, np.linalg.norm(line[1] - line[0])]), env.dt, k2)
            add('WALL', L, obs, act, line[-1], A=list(a), G=list(g), route_cells=L)
    # training windows
    ds = planner.experiment.dataset
    dobs, dact, lengths = ds.fields.observations, ds.fields.actions, ds.fields.path_lengths
    eps = [i for i, n in enumerate(lengths) if n >= H]
    rs = np.random.RandomState(0)
    for j in range(args.n_data):
        i = eps[rs.randint(len(eps))]
        s = rs.randint(lengths[i] - H + 1)
        obs, act = dobs[i, s:s + H].astype(np.float64), dact[i, s:s + H].astype(np.float64)
        add('DATA', j, obs, act, obs[-1, :2], episode=int(i), start=int(s))
    if args.smoke:
        return
    meta = dict(H=H, arrive=ARRIVE, t_levels=T_LEVELS, K=K, kp=args.kp, kv=args.kv, k2=k2, dt=float(env.dt),
                pick_seed=PICK_SEED, n_data=args.n_data)
    with open(os.path.join(out, 'scores.json'), 'w') as f:
        json.dump(dict(meta=meta, items=items), f)


if __name__ == '__main__':
    main()
