"""
Thin wrapper around the frozen pretrained Diffuser (shared by all experiments).

Implements the team interface  plan(state, goal, N) -> N trajectories  without touching
the model: it calls the original diffuser.guides.policies.Policy with the same goal
inpainting as scripts/plan_maze2d.py:  cond = {0: current_obs, H-1: [goal_x, goal_y, 0, 0]}.
"""
import contextlib
import io
import os
import random
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import diffuser.utils as utils
    import diffuser.datasets as datasets
    from diffuser.guides.policies import Policy


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_env(env_name):
    """Unwrapped d4rl env with .max_episode_steps (same as diffuser.datasets.load_environment)."""
    return datasets.load_environment(env_name)


class FrozenDiffuserPlanner:
    def __init__(self, env_name='maze2d-large-v1', loadpath='diffusion/H384_T256', epoch='latest',
                 device='cpu', logbase=None, threads=0):
        if threads and threads > 0:
            torch.set_num_threads(threads)
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            print(f'[planner] {device} requested but CUDA is not available: using cpu', flush=True)
            device = 'cpu'
        logbase = logbase or os.path.join(REPO, 'logs')
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            self.experiment = utils.load_diffusion(logbase, env_name, loadpath, epoch=epoch, device=device)
        self.load_time_s = time.perf_counter() - t0
        self.diffusion = self.experiment.ema
        self.policy = Policy(self.diffusion, self.experiment.dataset.normalizer)
        self.horizon = self.diffusion.horizon
        self.n_diffusion_steps = self.diffusion.n_timesteps
        self.epoch = self.experiment.epoch
        self.device = device

    def plan(self, obs, goal_xy, n=1, horizon=None, allow_longer=False):
        """Sample n plans from `obs` (4-d [x, y, vx, vy]) to goal_xy.

        horizon=None (or the trained horizon) uses the original Policy call, unchanged from B0.
        A shorter horizon runs the same sampler on a shorter sequence, with the goal inpainted at
        its last step. The U-Net is fully convolutional and downsamples len(dim_mults)-1 times, so
        the horizon only has to be a multiple of `horizon_multiple`; the model was not trained on it.
        allow_longer also accepts horizons above the trained one (diagnostics only).

        Returns (observations [n, H, 4], actions [n, H, 2], wall-clock seconds)."""
        h = self.horizon if horizon is None else int(horizon)
        cond = {0: np.asarray(obs, dtype=np.float64),
                h - 1: np.array([goal_xy[0], goal_xy[1], 0.0, 0.0])}
        t0 = time.perf_counter()
        if h == self.horizon:
            with contextlib.redirect_stdout(io.StringIO()):     # silence the sampling progress bar
                _, samples = self.policy(cond, batch_size=n)
            return samples.observations, samples.actions, time.perf_counter() - t0

        assert h % self.horizon_multiple == 0 and 2 * self.horizon_multiple <= h <= (h if allow_longer else self.horizon), \
            f'horizon {h} must be a multiple of {self.horizon_multiple} in [{2 * self.horizon_multiple}, {self.horizon}]'
        # Same steps as Policy.__call__, which cannot pass `horizon` through to the sampler.
        normalizer, action_dim = self.policy.normalizer, self.policy.action_dim
        conditions = self.policy._format_conditions(cond, n)
        sample = utils.to_np(self.diffusion(conditions, horizon=h, verbose=False))
        actions = normalizer.unnormalize(sample[:, :, :action_dim], 'actions')
        observations = normalizer.unnormalize(sample[:, :, action_dim:], 'observations')
        return observations, actions, time.perf_counter() - t0

    @property
    def horizon_multiple(self):
        """Sequence lengths the U-Net accepts: one stride-2 downsample per resolution except the last."""
        return 2 ** (len(self.diffusion.model.downs) - 1)

    def horizon_for(self, cells, steps_per_cell, min_horizon):
        """Planning horizon for a leg of `cells` maze cells, rounded up to a valid length.
        steps_per_cell and min_horizon come from the r3_horizon experiment."""
        m = self.horizon_multiple
        h = max(min_horizon, int(np.ceil(cells * steps_per_cell)))
        return int(min(self.horizon, m * int(np.ceil(h / m))))

    def meta(self):
        return dict(diffusion_epoch=self.epoch, horizon=self.horizon,
                    n_diffusion_steps=self.n_diffusion_steps, device=self.device,
                    torch_threads=torch.get_num_threads(), torch_version=torch.__version__,
                    load_time_s=self.load_time_s)


def waypoint_action(sequence, t, state):
    """The original plan_maze2d.py controller: track plan point t+1 (position + velocity error).
    After the plan is exhausted, steer to the last plan point with zero target velocity."""
    if t < len(sequence) - 1:
        next_waypoint = sequence[t + 1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
    return next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
