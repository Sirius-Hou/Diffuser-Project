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

    def plan(self, obs, goal_xy, n=1):
        """Sample n plans from `obs` (4-d [x, y, vx, vy]) to goal_xy.

        Returns (observations [n, H, 4], actions [n, H, 2], wall-clock seconds)."""
        cond = {0: np.asarray(obs, dtype=np.float64),
                self.horizon - 1: np.array([goal_xy[0], goal_xy[1], 0.0, 0.0])}
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):     # silence the sampling progress bar
            _, samples = self.policy(cond, batch_size=n)
        return samples.observations, samples.actions, time.perf_counter() - t0

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
