"""
Maze layouts as plain numpy arrays, so geometry / graph / collision code can be tested
WITHOUT importing d4rl or mujoco_py (e.g. on a Mac).

The string is copied from d4rl.pointmaze.maze_model.LARGE_MAZE (d4rl @ f2a05c0).
Runners that have the real env should call `check_matches_env(env)` once.
"""
import numpy as np

from experiments.common.maze_utils import WALL, EMPTY, GOAL

LARGE_MAZE = (
    "############\\"
    "#OOOO#OOOOO#\\"
    "#O##O#O#O#O#\\"
    "#OOOOOO#OOO#\\"
    "#O####O###O#\\"
    "#OO#O#OOOOO#\\"
    "##O#O#O#O###\\"
    "#OO#OOO#OGO#\\"
    "############"
)

LAYOUTS = {'maze2d-large-v1': LARGE_MAZE}
GOALS = {'maze2d-large-v1': (7.0, 9.0)}


def parse_maze(spec):
    """Same parsing as d4rl.pointmaze.maze_model.parse_maze: rows split by '\\'."""
    lines = spec.split('\\')
    H, W = len(lines), len(lines[0])
    arr = np.zeros((H, W), dtype=np.int32)
    for i, line in enumerate(lines):
        for j, ch in enumerate(line):
            arr[i, j] = WALL if ch == '#' else GOAL if ch == 'G' else EMPTY
    return arr


def load_maze_arr(env_name='maze2d-large-v1'):
    return parse_maze(LAYOUTS[env_name])


def check_matches_env(env, env_name='maze2d-large-v1'):
    """Assert the hard-coded layout equals the real env's maze (wall / free pattern)."""
    ours = load_maze_arr(env_name) == WALL
    theirs = np.asarray(env.maze_arr) == WALL
    assert ours.shape == theirs.shape and (ours == theirs).all(), 'hard-coded maze differs from env.maze_arr'
