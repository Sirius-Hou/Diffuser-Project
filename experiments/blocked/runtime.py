"""Builds (env, planner, maze, contact_fn) from a config: the real Diffuser + MuJoCo env,
or the fakes (cfg['fake'] = true / --fake) for logic tests without mujoco."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.mazes import check_matches_env  # noqa: E402


def build(cfg, fake=False):
    if fake or cfg.get('fake'):
        from experiments.common import fakes
        env = fakes.FakePointEnv(cfg.get('env', 'maze2d-large-v1'))
        planner = fakes.FakePlanner(env.maze_arr, detour_prob=cfg.get('fake_detour_prob', 0.3))
        return env, planner, env.maze_arr, fakes.fake_contact
    from experiments.common.planner import FrozenDiffuserPlanner, load_env
    planner = FrozenDiffuserPlanner(cfg.get('env', 'maze2d-large-v1'), cfg.get('diffusion_loadpath', 'diffusion/H384_T256'),
                                    cfg.get('diffusion_epoch', 'latest'), cfg.get('device', 'cpu'),
                                    threads=cfg.get('torch_threads', 0))
    env = load_env(cfg.get('env', 'maze2d-large-v1'))
    check_matches_env(env, cfg.get('env', 'maze2d-large-v1'))
    return env, planner, env.maze_arr, mu.wall_contact


def git_commit():
    import subprocess
    try:
        sha = subprocess.check_output(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.call(['git', '-C', REPO, 'diff', '--quiet', 'HEAD']) != 0
        return sha + ('+dirty' if dirty else '')
    except Exception:  # noqa: BLE001
        return 'unknown'
