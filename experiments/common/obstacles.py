"""
Mid-episode obstacles for D4RL maze2d (shared by all blocked-corridor experiments).

An obstacle is an axis-aligned rectangle in qpos / observation coordinates (same convention as
maze_utils). It is a virtual region: nothing is added to the MuJoCo model. The proposal counts
"entering it" as a collision, so the episode loop calls ball_hits() every step and ends the
episode on the first hit.

Team interface 2:  collides(trajectory, obstacle) -> bool
"""
import numpy as np

from . import maze_utils as mu


class Obstacle:
    def __init__(self, cell, rect, kind, t_appear=0):
        self.cell = (int(cell[0]), int(cell[1]))
        self.rect = tuple(float(v) for v in rect)        # (x_lo, x_hi, y_lo, y_hi)
        self.kind = kind                                 # 'full' | 'partial'
        self.t_appear = int(t_appear)

    @classmethod
    def full(cls, cell, t_appear=0):
        """Covers the whole cell, so a one-cell-wide corridor is completely blocked."""
        return cls(cell, (cell[0] + mu.CELL_LO, cell[0] + mu.CELL_HI,
                          cell[1] + mu.CELL_LO, cell[1] + mu.CELL_HI), 'full', t_appear)

    @classmethod
    def partial(cls, cell, travel_axis, side, frac=0.5, t_appear=0):
        """Covers `frac` of the corridor width in `cell`, over the cell's whole length.

        travel_axis: axis the corridor runs along (0 or 1); the obstacle is cut across the other axis.
        side: -1 covers the low-coordinate side of the width, +1 the high side.
        The free gap is (1 - frac) wide and must be wider than the ball."""
        assert 0.0 < frac < 1.0 and (1.0 - frac) > 2 * mu.BALL_RADIUS, 'gap must be wider than the ball'
        lo = [cell[0] + mu.CELL_LO, cell[1] + mu.CELL_LO]
        hi = [cell[0] + mu.CELL_HI, cell[1] + mu.CELL_HI]
        w = 1 - travel_axis
        if side < 0:
            hi[w] = lo[w] + frac
        else:
            lo[w] = hi[w] - frac
        return cls(cell, (lo[0], hi[0], lo[1], hi[1]), 'partial', t_appear)

    def gap_center(self):
        """Centre of the part of the cell the obstacle leaves free (the cell centre for 'full')."""
        c = mu.cell_center(self.cell)
        x_lo, x_hi, y_lo, y_hi = self.rect
        cell_lo = (self.cell[0] + mu.CELL_LO, self.cell[1] + mu.CELL_LO)
        cell_hi = (self.cell[0] + mu.CELL_HI, self.cell[1] + mu.CELL_HI)
        for axis, (lo, hi) in enumerate(((x_lo, x_hi), (y_lo, y_hi))):
            if lo > cell_lo[axis] + 1e-9:                # free strip on the low side
                c[axis] = 0.5 * (cell_lo[axis] + lo)
            elif hi < cell_hi[axis] - 1e-9:              # free strip on the high side
                c[axis] = 0.5 * (hi + cell_hi[axis])
        return c

    def distance(self, xys):
        """Distance from each point to the rectangle (0 inside). xys: [..., 2]."""
        xys = np.asarray(xys, dtype=np.float64)
        x_lo, x_hi, y_lo, y_hi = self.rect
        dx = np.maximum(np.maximum(x_lo - xys[..., 0], 0.0), xys[..., 0] - x_hi)
        dy = np.maximum(np.maximum(y_lo - xys[..., 1], 0.0), xys[..., 1] - y_hi)
        return np.hypot(dx, dy)

    def to_dict(self):
        return dict(cell=list(self.cell), rect=list(self.rect), kind=self.kind, t_appear=self.t_appear)

    @classmethod
    def from_dict(cls, d):
        return cls(d['cell'], d['rect'], d['kind'], d['t_appear'])


def ball_hits(xy, obstacle):
    """True if the ball (a disc of BALL_RADIUS at xy) overlaps the obstacle."""
    return bool(obstacle.distance(xy) < mu.BALL_RADIUS)


def first_collision(traj_xy, obstacle, margin=0.0):
    """Index of the first trajectory point whose ball (radius + margin) overlaps the obstacle, or None.

    Consecutive points are also checked at their midpoint, so a plan cannot step over a thin obstacle."""
    xy = np.asarray(traj_xy, dtype=np.float64)[:, :2]
    if len(xy) == 0:
        return None
    hit = obstacle.distance(xy) < mu.BALL_RADIUS + margin
    if len(xy) > 1:
        mid_hit = obstacle.distance(0.5 * (xy[1:] + xy[:-1])) < mu.BALL_RADIUS + margin
        hit[1:] |= mid_hit
    idx = np.flatnonzero(hit)
    return int(idx[0]) if len(idx) else None


def collides(traj_xy, obstacle, margin=0.0):
    return first_collision(traj_xy, obstacle, margin) is not None


def corridor_axis(maze_arr, cell):
    """Axis a straight corridor runs along at `cell` (0 or 1), or None for corners, junctions and dead ends."""
    free = [mu.is_free(maze_arr, (cell[0] + d[0], cell[1] + d[1])) for d in mu.NEIGHBORS4]
    if free == [True, True, False, False]:
        return 0
    if free == [False, False, True, True]:
        return 1
    return None


def draw_obstacle(ax, obstacle, color='tab:red'):
    """Draw on an axis set up by maze_utils.draw_maze (x = qpos[1], y = qpos[0])."""
    from matplotlib.patches import Rectangle
    x_lo, x_hi, y_lo, y_hi = obstacle.rect
    ax.add_patch(Rectangle((y_lo, x_lo), y_hi - y_lo, x_hi - x_lo, color=color, alpha=0.6, zorder=4))
