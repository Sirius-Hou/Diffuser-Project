"""
Mid-episode obstacles (shared by every method; numpy only, no mujoco).

Setting (proposal §1): an obstacle covers all or part of ONE maze cell, appears at step
`appear_t`, then stays fixed. Its position is known as soon as it appears (perfect sensing).
Entering it counts as a collision. The obstacle is purely logical: it is NOT added to MuJoCo,
so the runner must check the ball every step and end the episode on contact.

Geometry follows maze_utils: in observation coordinates cell (i, j) spans
[i-0.7, i+0.3] x [j-0.7, j+0.3]; the ball is a disc of radius 0.1.

Obstacle kinds
  full     the whole cell.
  partial  a slab across the corridor covering a fraction `frac` of the corridor width,
           attached to one side ('lo' or 'hi' along the cross-corridor axis), full length
           along the corridor. The free gap is (1 - frac). It is passable for the ball if the
           gap > 2 * BALL_RADIUS.
"""
from dataclasses import dataclass, asdict

import numpy as np

from experiments.common import maze_utils as mu


@dataclass(frozen=True)
class Obstacle:
    cell: tuple                 # (i, j) maze cell
    appear_t: int               # control step at which it appears (known from then on)
    kind: str = 'full'          # 'full' | 'partial'
    frac: float = 1.0           # partial only: covered fraction of the corridor width
    side: str = 'lo'            # partial only: which wall the slab is attached to
    axis: int = 1               # partial only: cross-corridor axis (0 = x/i, 1 = y/j)

    def rect(self):
        """Axis-aligned box ((x0, x1), (y0, y1)) in observation coordinates."""
        i, j = self.cell
        lo = [i + mu.CELL_LO, j + mu.CELL_LO]
        hi = [i + mu.CELL_HI, j + mu.CELL_HI]
        if self.kind == 'partial':
            a = self.axis
            if self.side == 'lo':
                hi[a] = lo[a] + self.frac
            else:
                lo[a] = hi[a] - self.frac
        return (lo[0], hi[0]), (lo[1], hi[1])

    @property
    def blocks_cell(self):
        """True if the ball cannot pass through this cell at all (used to cut the graph)."""
        return self.kind == 'full' or (1.0 - self.frac) <= 2 * mu.BALL_RADIUS

    def active(self, t):
        return t >= self.appear_t

    def to_dict(self):
        d = asdict(self)
        d['cell'] = list(self.cell)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d['cell'] = tuple(d['cell'])
        return Obstacle(**d)


def corridor_axis(maze_arr, cell):
    """Cross-corridor axis of a corridor cell: 1 if the corridor runs along x (free above/below),
    0 if it runs along y (free left/right). None for junctions, corners and dead ends."""
    i, j = cell
    up_down = mu.is_free(maze_arr, (i - 1, j)) and mu.is_free(maze_arr, (i + 1, j))
    left_right = mu.is_free(maze_arr, (i, j - 1)) and mu.is_free(maze_arr, (i, j + 1))
    n_free = sum(mu.is_free(maze_arr, (i + di, j + dj)) for di, dj in mu.NEIGHBORS4)
    if n_free != 2:
        return None
    if up_down:
        return 1
    if left_right:
        return 0
    return None


def disc_hits_rect(xy, rect, radius):
    """True if a disc of `radius` at xy overlaps the box (touching counts when radius == 0)."""
    (x0, x1), (y0, y1) = rect
    dx = max(x0 - xy[0], 0.0, xy[0] - x1)
    dy = max(y0 - xy[1], 0.0, xy[1] - y1)
    if radius <= 0:
        return dx == 0.0 and dy == 0.0
    return dx * dx + dy * dy < radius * radius


def points_hit(xys, obstacle, radius):
    """Boolean mask over trajectory points: does a disc of `radius` around each point overlap?"""
    xys = np.asarray(xys, dtype=np.float64)[:, :2]
    (x0, x1), (y0, y1) = obstacle.rect()
    dx = np.maximum.reduce([x0 - xys[:, 0], np.zeros(len(xys)), xys[:, 0] - x1])
    dy = np.maximum.reduce([y0 - xys[:, 1], np.zeros(len(xys)), xys[:, 1] - y1])
    if radius <= 0:
        return (dx == 0) & (dy == 0)
    return dx * dx + dy * dy < radius * radius


def first_collision(xys, obstacles, t=None, inflate=mu.BALL_RADIUS):
    """Team interface `會不會撞(軌跡, 障礙物)`.

    xys       plan points (n, >=2); only x, y are used. Walls are NOT checked here.
    obstacles iterable of Obstacle; if t is given only those active at step t count.
    inflate   ball radius plus any safety margin.
    Returns the index of the first colliding point, or None if the trajectory is clear.
    """
    xys = np.asarray(xys)
    if len(xys) == 0:
        return None
    first = None
    for ob in obstacles:
        if t is not None and not ob.active(t):
            continue
        hit = np.flatnonzero(points_hit(xys, ob, inflate))
        if len(hit) and (first is None or hit[0] < first):
            first = int(hit[0])
    return first


def collides(xys, obstacles, t=None, inflate=mu.BALL_RADIUS):
    return first_collision(xys, obstacles, t, inflate) is not None


def ball_collides(xy, obstacles, t):
    """Executed-ball check used by the runner every step (exact ball radius, no margin)."""
    return any(ob.active(t) and disc_hits_rect(xy, ob.rect(), mu.BALL_RADIUS) for ob in obstacles)


def blocked_cells(obstacles, t=None):
    """Cells to delete from the maze graph (only obstacles that leave no passable gap)."""
    return {ob.cell for ob in obstacles if ob.blocks_cell and (t is None or ob.active(t))}


def draw_obstacles(ax, obstacles, color='tab:purple', alpha=0.85):
    """Draw on a draw_maze axis (x = qpos[1], y = qpos[0])."""
    from matplotlib.patches import Rectangle
    for ob in obstacles:
        (x0, x1), (y0, y1) = ob.rect()
        ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, color=color, alpha=alpha, zorder=3))
