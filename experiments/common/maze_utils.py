"""
Maze geometry helpers for D4RL maze2d (shared by all experiments).

Coordinate convention (verified with MuJoCo contacts, see check_geometry.py):
    d4rl point_maze places wall cell (i, j) at world (i+1, j+1) with half-size 0.5, and the
    particle body at world (1.2, 1.2) + qpos. So in qpos / observation coordinates, maze cell
    (i, j) covers [i-0.7, i+0.3] x [j-0.7, j+0.3]. Cell index of a point = round(q + 0.2).
    The goal (7, 9) lies inside cell (7, 9). The ball radius is 0.1.
"""
from collections import deque

import numpy as np

WALL, EMPTY, GOAL = 10, 11, 12          # d4rl.pointmaze.maze_model tile codes
BALL_RADIUS = 0.1
CELL_OFFSET = 0.2
CELL_LO, CELL_HI = -0.7, 0.3            # cell (i,j) spans [i+CELL_LO, i+CELL_HI] per axis

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def xy_to_cell(xy):
    return int(np.floor(xy[0] + CELL_OFFSET + 0.5)), int(np.floor(xy[1] + CELL_OFFSET + 0.5))


def cell_center(cell):
    """qpos coordinates of the geometric centre of a cell."""
    return np.array([cell[0] - CELL_OFFSET, cell[1] - CELL_OFFSET])


def is_free(maze_arr, cell):
    W, H = maze_arr.shape
    return 0 <= cell[0] < W and 0 <= cell[1] < H and maze_arr[cell] != WALL


def free_cells(maze_arr):
    return [tuple(c) for c in np.argwhere(maze_arr != WALL)]


def point_in_wall(maze_arr, xy, inflate=0.0):
    """True if the point (or a disc of radius `inflate` around it) overlaps a wall cell."""
    w, h = xy_to_cell(xy)
    W, H = maze_arr.shape
    for dw in (-1, 0, 1):
        for dh in (-1, 0, 1):
            cw, ch = w + dw, h + dh
            if not (0 <= cw < W and 0 <= ch < H) or maze_arr[cw, ch] != WALL:
                continue
            dx = max(cw + CELL_LO - xy[0], 0.0, xy[0] - (cw + CELL_HI))
            dy = max(ch + CELL_LO - xy[1], 0.0, xy[1] - (ch + CELL_HI))
            if inflate == 0.0:
                if dx == 0.0 and dy == 0.0:
                    return True
            elif np.hypot(dx, dy) < inflate:
                return True
    return False


def count_points_in_wall(maze_arr, xys, inflate=0.0):
    return int(sum(point_in_wall(maze_arr, p, inflate) for p in xys))


def bfs_path(maze_arr, start, goal, blocked=()):
    """Shortest 4-connected path of cells from start to goal (inclusive), or None.
    `blocked` is a collection of extra cells treated as walls (used later for obstacles)."""
    blocked = set(map(tuple, blocked))
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        for d in NEIGHBORS4:
            n = (c[0] + d[0], c[1] + d[1])
            if is_free(maze_arr, n) and n not in blocked and n not in prev:
                prev[n] = c
                q.append(n)
    if goal not in prev:
        return None
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev[c]
    return path[::-1]


def cell_sequence(xys):
    """Cells visited by a trajectory, consecutive duplicates removed."""
    seq = []
    for xy in xys:
        c = xy_to_cell(xy)
        if not seq or seq[-1] != c:
            seq.append(c)
    return seq


def path_length(xys):
    xys = np.asarray(xys)
    return float(np.linalg.norm(np.diff(xys, axis=0), axis=1).sum()) if len(xys) > 1 else 0.0


def wall_contact(env):
    """True if the MuJoCo particle currently touches any maze wall geom."""
    m, d = env.sim.model, env.sim.data
    for i in range(d.ncon):
        c = d.contact[i]
        n1, n2 = m.geom_id2name(c.geom1), m.geom_id2name(c.geom2)
        if (n1 == 'particle_geom' and n2 and n2.startswith('wall')) or \
           (n2 == 'particle_geom' and n1 and n1.startswith('wall')):
            return True
    return False


def contact_events(flags):
    """Number of separate contact episodes (rising edges) in a boolean per-step array."""
    flags = np.asarray(flags, dtype=bool)
    if len(flags) == 0:
        return 0
    return int(flags[0]) + int(np.sum(flags[1:] & ~flags[:-1]))


def draw_maze(ax, maze_arr, color='0.35'):
    """Draw walls on a matplotlib axis with x = qpos[1], y = qpos[0] (Diffuser renderer orientation)."""
    from matplotlib.patches import Rectangle
    W, H = maze_arr.shape
    for w in range(W):
        for h in range(H):
            if maze_arr[w, h] == WALL:
                ax.add_patch(Rectangle((h + CELL_LO, w + CELL_LO), 1, 1, color=color))
    ax.set_xlim(CELL_LO, H + CELL_LO)
    ax.set_ylim(W + CELL_LO, CELL_LO)
    ax.set_aspect('equal')
