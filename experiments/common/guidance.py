"""
Test-time guidance for the frozen Diffuser with an analytic obstacle cost (no training).

The Diffuser paper (Janner et al. 2022, Sec. 3.2 / Alg. 1) biases each reverse step with the gradient of a
guide J; the paper learns J, we write it by hand: a squared hinge on the signed distance from each plan
point to the obstacle rectangle (optionally also to the wall cells). The maze2d branch has no guided
sampler, so `guided_sample` re-implements GaussianDiffusion.p_sample_loop step for step and adds one push:

  variant 'x0' : predict x0 (the model predicts x0 directly), move its positions down the cost gradient in
                 maze coordinates for "time" scale (ceil(scale / max_lr) gradient steps of at most max_lr, since
                 one big step diverges on the stiff step-length term), then take the usual posterior step.
  variant 'xt' : as the original locomotion code (diffuser/sampling/functions.py on main): before the
                 model call, push the noisy x_t by scale * posterior_variance * grad, n_guide_steps times,
                 no push for t < t_stopgrad.

With scale 0 the loop draws the same random numbers in the same order as p_sample_loop, so its output
equals planner.plan (checked by diag/guided_sampling.py --check).

`via` (proposal option b, interior inpainting): {plan index: xy}. Those plan points have their x-y fixed after
every reverse step, as the start and goal are; their velocity is left to the model. No extra random numbers.
"""
import numpy as np
import torch

from diffuser.models.helpers import apply_conditioning, extract

from . import maze_utils as mu

WINDOWS = ('all', 'early', 'late')


def _signed_distance(p, rects):
    """Signed distance from points p [..., 2] to each rectangle rects [R, 4] (x_lo, x_hi, y_lo, y_hi):
    positive outside, negative inside (depth to the nearest side). Returns [..., R]."""
    x, y = p[..., 0:1], p[..., 1:2]
    x_lo, x_hi, y_lo, y_hi = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    dx = torch.relu(x_lo - x) + torch.relu(x - x_hi)
    dy = torch.relu(y_lo - y) + torch.relu(y - y_hi)
    outside = torch.sqrt(dx * dx + dy * dy + 1e-12)
    inside = -torch.min(torch.min(x - x_lo, x_hi - x), torch.min(y - y_lo, y_hi - y))
    return torch.where((dx > 0) | (dy > 0), outside, inside)


def _signed_distance_each(p, rect):
    """Signed distance from each point p [..., 2] to its own rectangle rect [..., 4]. Returns [...]."""
    x, y = p[..., 0], p[..., 1]
    x_lo, x_hi, y_lo, y_hi = rect.unbind(-1)
    dx = torch.relu(x_lo - x) + torch.relu(x - x_hi)
    dy = torch.relu(y_lo - y) + torch.relu(y - y_hi)
    outside = torch.sqrt(dx * dx + dy * dy + 1e-12)
    inside = -torch.min(torch.min(x - x_lo, x_hi - x), torch.min(y - y_lo, y_hi - y))
    return torch.where((dx > 0) | (dy > 0), outside, inside)


class Cost:
    """sum over plan points of relu(BALL_RADIUS + margin - sd)^2 for the obstacle, plus optionally
      walls  : wall_weight * relu(wall_margin - sd_wall)^2 (a point counts as in a wall when its centre is in a wall cell)
      smooth : smooth_weight * relu(|p[k+1] - p[k]| - max_step)^2, so the plan cannot jump over the obstacle"""

    def __init__(self, obstacle, margin=0.15, walls=None, wall_margin=0.0, wall_weight=1.0,
                 smooth_weight=0.0, max_step=0.15, device='cpu'):
        self.obs = torch.tensor([obstacle.rect], dtype=torch.float32, device=device)
        self.margin = mu.BALL_RADIUS + margin
        # walls: the maze array; only the 3x3 cells around each point are checked
        self.walls = None if walls is None else torch.tensor(walls == mu.WALL, device=device)
        self.wall_margin, self.wall_weight = wall_margin, wall_weight
        self.smooth_weight, self.max_step = smooth_weight, max_step

    def _wall_cost(self, p):
        W, H = self.walls.shape
        base = torch.floor(p.detach() - mu.CELL_LO).long()          # cell (i, j) spans [i + CELL_LO, i + CELL_HI]
        total = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ci, cj = base[..., 0] + di, base[..., 1] + dj
                inside = (ci >= 0) & (ci < W) & (cj >= 0) & (cj < H)
                wall = torch.ones_like(inside)                       # outside the array counts as wall
                wall[inside] = self.walls[ci[inside], cj[inside]]
                x_lo, y_lo = ci + mu.CELL_LO, cj + mu.CELL_LO
                rect = torch.stack([x_lo, x_lo + 1, y_lo, y_lo + 1], dim=-1).float()
                sd = _signed_distance_each(p, rect)
                total = total + (wall.float() * torch.relu(self.wall_margin - sd).pow(2)).sum(-1)
        return total

    def __call__(self, p):
        """p [B, H, 2] in maze coordinates -> cost per plan [B]."""
        c = torch.relu(self.margin - _signed_distance(p, self.obs)).pow(2).sum(dim=(-1, -2))
        if self.walls is not None:
            c = c + self.wall_weight * self._wall_cost(p)
        if self.smooth_weight:
            step = torch.sqrt(((p[:, 1:] - p[:, :-1]) ** 2).sum(-1) + 1e-12)
            c = c + self.smooth_weight * torch.relu(step - self.max_step).pow(2).sum(-1)
        return c


def in_window(window, i, n_timesteps):
    if window == 'all':
        return True
    if window == 'early':
        return i >= n_timesteps // 2
    if window == 'late':
        return i < n_timesteps // 2
    raise ValueError(window)


def guided_sample(planner, obs, goal_xy, n, cost=None, scale=0.0, window='all', variant='x0',
                  horizon=None, n_guide_steps=2, t_stopgrad=2, max_lr=0.1, via=None):
    """Like planner.plan(obs, goal_xy, n, horizon) but with guidance. Returns (observations, actions, info)
    with info = dict(push_mean=mean |push| per guided reverse step in maze units)."""
    diffusion, policy = planner.diffusion, planner.policy
    action_dim = policy.action_dim
    obs_norm = policy.normalizer.normalizers['observations']
    lo = torch.tensor(obs_norm.mins[:2], dtype=torch.float32, device=planner.device)
    span = torch.tensor(obs_norm.maxs[:2] - obs_norm.mins[:2], dtype=torch.float32, device=planner.device)
    sl = slice(action_dim, action_dim + 2)                       # position columns of [ax, ay, x, y, vx, vy]

    def to_maze(z):
        return (z + 1) / 2 * span + lo

    def to_norm(p):
        return 2 * (p - lo) / span - 1

    h = planner.horizon if horizon is None else int(horizon)
    cond = policy._format_conditions({0: np.asarray(obs, dtype=np.float64),
                                      h - 1: np.array([goal_xy[0], goal_xy[1], 0.0, 0.0])}, n)
    via = {int(k): to_norm(torch.tensor(np.asarray(xy, dtype=np.float32), device=planner.device))
           for k, xy in (via or {}).items()}

    def condition(x):
        x = apply_conditioning(x, cond, action_dim)
        for k, xy in via.items():
            x[:, k, sl] = xy
        return x

    T = diffusion.n_timesteps
    guide = cost is not None and scale > 0
    pushes = []

    x = torch.randn((n, h, diffusion.transition_dim), device=planner.device)
    x = condition(x)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=planner.device, dtype=torch.long)
        on = guide and in_window(window, i, T)
        if on and variant == 'xt' and i >= t_stopgrad:
            var = extract(diffusion.posterior_log_variance_clipped, t, x.shape).exp()
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    z = x[..., sl].detach().requires_grad_()
                    g = torch.autograd.grad(cost(to_maze(z)).sum(), z)[0]
                step = scale * var[..., :1] * g
                x = x.clone()
                x[..., sl] = x[..., sl] - step
                x = condition(x)
                pushes.append(float((step * span / 2).norm(dim=-1).mean()))
        with torch.no_grad():
            x0 = diffusion.predict_start_from_noise(x, t=t, noise=diffusion.model(x, cond, t))
            if diffusion.clip_denoised:
                x0.clamp_(-1., 1.)
        if on and variant == 'x0':
            p0 = to_maze(x0[..., sl]).detach()
            p2 = p0.clone()
            n_inner = max(1, int(np.ceil(scale / max_lr)))     # gradient flow for time `scale`, stable steps
            for _ in range(n_inner):
                with torch.enable_grad():
                    p = p2.requires_grad_()
                    g = torch.autograd.grad(cost(p).sum(), p)[0]
                p2 = p.detach() - (scale / n_inner) * g
            pushes.append(float((p2 - p0).norm(dim=-1).mean()))
            x0 = x0.clone()
            x0[..., sl] = to_norm(p2).clamp(-1., 1.)
        with torch.no_grad():
            mean, _, logvar = diffusion.q_posterior(x_start=x0, x_t=x, t=t)
            noise = torch.randn_like(x)
            nonzero = (1 - (t == 0).float()).reshape(n, 1, 1)
            x = mean + nonzero * (0.5 * logvar).exp() * noise
            x = condition(x)

    sample = x.detach().cpu().numpy()
    actions = policy.normalizer.unnormalize(sample[:, :, :action_dim], 'actions')
    observations = policy.normalizer.unnormalize(sample[:, :, action_dim:], 'observations')
    return observations, actions, dict(push_mean=float(np.mean(pushes)) if pushes else 0.0)
