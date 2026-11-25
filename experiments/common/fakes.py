"""
Stand-ins for the MuJoCo env and the Diffuser, ONLY for testing the loop logic on a machine
without mujoco / weights (e.g. a Mac). Numbers produced with these mean nothing.

FakePointEnv   point mass on the maze grid; walls stop the ball; same interface as the d4rl env
               subset used by rollout.py.
FakePlanner    obstacle-unaware like Diffuser: a route through the free cells (usually a
               shortest one, sometimes via a random detour cell), stretched over the horizon so it
               arrives ~24 steps before the end (like Diffuser, B0 README F1), plus small jitter.
"""
import numpy as np

from experiments.common import maze_utils as mu
from experiments.common.mazes import load_maze_arr, GOALS


class FakePointEnv:
    def __init__(self, env_name='maze2d-large-v1', max_episode_steps=800):
        self.name = env_name
        self.maze_arr = load_maze_arr(env_name)
        self._target = np.array(GOALS[env_name])
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.RandomState(0)
        self.q = np.zeros(2)
        self.v = np.zeros(2)
        self._contact = False

    def seed(self, s):
        self.rng = np.random.RandomState(s)

    def reset(self):
        cells = [c for c in mu.free_cells(self.maze_arr) if tuple(c) != mu.xy_to_cell(self._target)]
        c = cells[self.rng.randint(len(cells))]
        self.q = mu.cell_center(c) + self.rng.uniform(-0.1, 0.1, 2)
        self.v = self.rng.normal(0, 0.05, 2)
        return self.state_vector()

    def state_vector(self):
        return np.concatenate([self.q, self.v])

    def step(self, a):
        a = np.clip(a, -1, 1)
        v = 0.6 * self.v + 0.05 * a * 10
        q = self.q + 0.1 * v
        self._contact = mu.point_in_wall(self.maze_arr, q, mu.BALL_RADIUS)
        if self._contact:                       # slide along the wall: keep the free axis
            for ax in (0, 1):
                trial = self.q.copy()
                trial[ax] = q[ax]
                if mu.point_in_wall(self.maze_arr, trial, mu.BALL_RADIUS):
                    v[ax] = 0.0
                    q[ax] = self.q[ax]
        self.q, self.v = q, v
        return self.state_vector(), 0.0, False, {}


def fake_contact(env):
    return env._contact


class FakePlanner:
    horizon = 384
    n_diffusion_steps = 0
    epoch = 'fake'
    device = 'cpu'

    def __init__(self, maze_arr, detour_prob=0.3, seed=0):
        self.maze = maze_arr
        self.detour_prob = detour_prob
        # uses the global numpy RNG, which rollout.start_episode seeds per episode, so a given
        # seed gives the same plans every time (like torch.manual_seed for the real model)
        self.rng = np.random

    def _route(self, a, b):
        if self.rng.rand() < self.detour_prob:
            free = mu.free_cells(self.maze)
            via = tuple(free[self.rng.randint(len(free))])
            p1, p2 = mu.bfs_path(self.maze, a, via), mu.bfs_path(self.maze, via, b)
            if p1 and p2:
                return p1 + p2[1:]
        return mu.bfs_path(self.maze, a, b)

    def plan(self, obs, goal_xy, n=1, horizon=None, via=None):
        H = int(horizon or self.horizon)
        if via:                                   # fake middle inpainting: two stitched pieces
            (k, xy), = via
            a, _, _ = self.plan(obs, xy, n=n, horizon=k + 24)
            out = []
            for p in a:
                p = p[:k + 1]
                b, _, _ = self.plan(np.concatenate([p[-1, :2], [0, 0]]), goal_xy, n=1, horizon=H - k + 23)
                out.append(np.vstack([p, b[0][1:H - k]]))
            return np.array(out), np.zeros((n, H, 2)), 0.0
        a, b = mu.xy_to_cell(obs[:2]), mu.xy_to_cell(goal_xy)
        out = []
        for _ in range(n):
            cells = self._route(a, b) or [a, b]
            mid = [mu.cell_center(c) for c in cells[1:-1]]
            pts = np.vstack([obs[:2]] + mid + [goal_xy])
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            arrive = max(H - 24, 2)
            u = np.minimum(np.arange(H) / arrive, 1.0) * s[-1]
            xy = np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1)
            xy[1:-1] += self.rng.normal(0, 0.01, (H - 2, 2))
            v = np.vstack([np.diff(xy, axis=0) / 0.1 * 0.1, [[0, 0]]])
            out.append(np.hstack([xy, v]))
        return np.array(out), np.zeros((n, H, 2)), 0.0
