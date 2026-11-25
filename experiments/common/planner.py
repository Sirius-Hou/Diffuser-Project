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

    def plan(self, obs, goal_xy, n=1, horizon=None, via=None):
        """Sample n plans from `obs` (4-d [x, y, vx, vy]) to goal_xy.

        horizon  plan length H (default: the trained 384). Must be a multiple of 4 (the U-Net
                 down-samples twice). H != 384 was never seen in training; its quality must be
                 measured, not assumed. With the default the call is exactly the original
                 Policy call, so B0 results are unchanged.
        via      optional list of (k, (x, y)): additionally pin ONLY the x-y position of plan
                 step k (middle inpainting, proposal contribution 3). Velocity at k is left free.
                 The model never saw such conditions in training.
        Returns (observations [n, H, 4], actions [n, H, 2], wall-clock seconds)."""
        H = int(horizon or self.horizon)
        assert H % 4 == 0 and H >= 8, f'horizon must be a multiple of 4, got {H}'
        cond = {0: np.asarray(obs, dtype=np.float64),
                H - 1: np.array([goal_xy[0], goal_xy[1], 0.0, 0.0])}
        t0 = time.perf_counter()
        if via:
            return self._plan_via(cond, via, n, H, t0=time.perf_counter())
        with contextlib.redirect_stdout(io.StringIO()):     # silence the sampling progress bar
            if H == self.horizon:
                _, samples = self.policy(cond, batch_size=n)
                obs_out, act_out = samples.observations, samples.actions
            else:
                # Same steps as Policy.__call__, but passing horizon to the (unchanged) sampler.
                conditions = self.policy._format_conditions(cond, n)
                sample = utils.to_np(self.diffusion(conditions, horizon=H))
                a_dim = self.policy.action_dim
                act_out = self.policy.normalizer.unnormalize(sample[:, :, :a_dim], 'actions')
                obs_out = self.policy.normalizer.unnormalize(sample[:, :, a_dim:], 'observations')
        return obs_out, act_out, time.perf_counter() - t0

    def _plan_via(self, cond, via, n, H, t0):
        """Sampling with extra x-y-only conditions. The original apply_conditioning (which pins
        the full [x, y, vx, vy] observation) is swapped for one that also pins x-y at the `via`
        steps, only for the duration of this call; model weights and code are unchanged."""
        import diffuser.models.diffusion as dmod
        a_dim = self.policy.action_dim
        conditions = self.policy._format_conditions(cond, n)
        pins = []
        for k, xy in via:
            k = int(k)
            assert 0 < k < H - 1, f'via step must be inside the plan, got {k} for H={H}'
            full = self.policy.normalizer.normalize(np.array([xy[0], xy[1], 0.0, 0.0]), 'observations')
            pins.append((k, torch.tensor(full[:2], dtype=torch.float32, device=self.device)))
        original = dmod.apply_conditioning

        def apply_with_via(x, c, action_dim):
            x = original(x, c, action_dim)
            for k, v in pins:
                x[:, k, action_dim:action_dim + 2] = v
            return x

        dmod.apply_conditioning = apply_with_via
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sample = utils.to_np(self.diffusion(conditions, horizon=H))
        finally:
            dmod.apply_conditioning = original
        act_out = self.policy.normalizer.unnormalize(sample[:, :, :a_dim], 'actions')
        obs_out = self.policy.normalizer.unnormalize(sample[:, :, a_dim:], 'observations')
        return obs_out, act_out, time.perf_counter() - t0

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
