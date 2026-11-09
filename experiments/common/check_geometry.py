"""
Verify the qpos <-> maze-cell convention used in maze_utils.py with real MuJoCo contacts.

From cell (1,1) the ball is pushed in each of the 4 directions until it touches a wall.
The contact position must equal the wall face predicted by maze_utils, plus the ball radius.

    bash experiments/common/env/run.sh python experiments/common/check_geometry.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import load_env  # noqa: E402

env = load_env('maze2d-large-v1')
env.seed(0)
env.reset()
maze = env.maze_arr
ok = True
for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    env.set_state(np.array([1.0, 1.0]), np.zeros(2))
    for t in range(400):
        env.step(np.array(d) * 0.3)
        if mu.wall_contact(env):
            break
    q = env.sim.data.qpos.copy()
    # walk the grid from (1,1) along d to the first wall cell, predict where the ball stops
    c = (1, 1)
    while mu.is_free(maze, (c[0] + d[0], c[1] + d[1])):
        c = (c[0] + d[0], c[1] + d[1])
    axis = 0 if d[0] else 1
    face = (c[axis] + mu.CELL_HI) if (d[0] + d[1]) > 0 else (c[axis] + mu.CELL_LO)
    pred = face - np.sign(d[0] + d[1]) * mu.BALL_RADIUS
    err = abs(q[axis] - pred)
    ok &= err < 0.05
    print(f'dir {d}: contact at qpos {q.round(3)} | predicted {pred:.3f} on axis {axis} | err {err:.3f}')
print('GEOMETRY OK' if ok else 'GEOMETRY MISMATCH')
sys.exit(0 if ok else 1)
