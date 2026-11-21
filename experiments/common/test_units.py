"""
Unit tests for the model-free helpers (maze_graph, obstacles). Needs numpy only, no MuJoCo or torch:

    python experiments/common/test_units.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.common import maze_graph as mg  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402

# d4rl.pointmaze.maze_model.LARGE_MAZE (maze2d-large-v1); env.maze_arr[i, j] is row i, column j
LARGE_MAZE = ["############",
              "#OOOO#OOOOO#",
              "#O##O#O#O#O#",
              "#OOOOOO#OOO#",
              "#O####O###O#",
              "#OO#O#OOOOO#",
              "##O#O#O#O###",
              "#OO#OOO#OGO#",
              "############"]
CODE = {'#': mu.WALL, 'O': mu.EMPTY, 'G': mu.GOAL}
MAZE = np.array([[CODE[ch] for ch in row] for row in LARGE_MAZE])
GOAL = (7, 9)


def test_astar_matches_bfs():
    cells = mu.free_cells(MAZE)
    n = 0
    for s in cells:
        for b in [()] + [(c,) for c in cells if c not in (s, GOAL)]:
            a, r = mg.astar(MAZE, s, GOAL, b), mu.bfs_path(MAZE, s, GOAL, b)
            assert (a is None) == (r is None), (s, b)
            if a is None:
                continue
            assert len(a) == len(r) and a[0] == s and a[-1] == GOAL, (s, b)
            assert all(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for p, q in zip(a, a[1:]))
            assert not set(a) & set(b)
            n += 1
    return f'{n} (start, blocked) cases'


def test_detour_counts():
    """Most single-cell blockages leave no detour (proposal); the known long detour is found."""
    cells = mu.free_cells(MAZE)
    with_detour = total = 0
    for s in cells:
        base = mu.bfs_path(MAZE, s, GOAL)
        for c in base[1:-1]:
            total += 1
            with_detour += mg.astar(MAZE, s, GOAL, [c]) is not None
    p = mg.astar(MAZE, (3, 6), GOAL, [(5, 7)])
    assert len(mu.bfs_path(MAZE, (3, 6), GOAL)) - 1 == 7 and len(p) - 1 == 15, len(p)
    return f'{with_detour}/{total} blockages on a shortest path leave a detour'


def test_waypoints():
    p = mg.astar(MAZE, (4, 6), GOAL, [(5, 7)])           # ball in the corridor below junction (3,6)
    assert p[1] == (3, 6), p
    wp = mg.waypoints(MAZE, p, [(5, 7)])
    assert p[wp[-1]] == GOAL and wp == sorted(set(wp)) and 0 not in wp
    assert all(mg.degree(MAZE, p[i], [(5, 7)]) >= 3 for i in wp[:-1])
    assert (3, 6) in [p[i] for i in wp]
    short = mg.waypoints(MAZE, p, [(5, 7)], max_leg=3)
    assert set(wp) <= set(short) and max(mg.leg_lengths(short)) <= 3
    assert sum(mg.leg_lengths(short)) == len(p) - 1
    turn = mg.waypoints(MAZE, p, [(5, 7)], corners=True)
    assert set(wp) <= set(turn) and all(mg.is_corner(p, i) for i in set(turn[:-1]) - set(wp))
    prev = 0
    for i in turn:                                       # every leg is a straight line of cells
        assert len({(b[0] - a[0], b[1] - a[1]) for a, b in zip(p[prev:i], p[prev + 1:i + 1])}) == 1
        prev = i
    return f'{[p[i] for i in wp]} | max_leg=3 -> {len(short)} waypoints | corners -> {len(turn)}'


def test_sparse_waypoints():
    blocked = [(5, 7)]
    p = mg.astar(MAZE, (5, 2), GOAL, blocked)
    wp = mg.sparse_waypoints(MAZE, p, blocked)
    assert p[wp[-1]] == GOAL and wp == sorted(set(wp))
    prev = 0
    for i in wp:                                         # each leg's natural shortest route avoids the cell
        assert (5, 7) not in mu.bfs_path(MAZE, p[prev], p[i])
        prev = i
    capped = mg.sparse_waypoints(MAZE, p, blocked, max_leg=4)
    assert max(mg.leg_lengths(capped)) <= 4 and len(capped) >= len(wp)
    assert mg.sparse_waypoints(MAZE, mu.bfs_path(MAZE, (5, 2), GOAL), []) == [len(mu.bfs_path(MAZE, (5, 2), GOAL)) - 1]
    return f'{[p[i] for i in wp]} ({len(p) - 1} cells)'


def test_line_plan():
    dt = 0.01
    p = mg.line_plan([(0.0, 0.0), (0.0, 2.0)], 65, dt)
    assert p.shape == (65, 4) and np.allclose(p[0, :2], (0, 0)) and np.allclose(p[-1, :2], (0, 2))
    step = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    assert np.allclose(step, 2 / 64)                     # constant speed
    assert np.allclose(p[:-1, 2:], np.diff(p[:, :2], axis=0) / dt) and np.allclose(p[-1, 2:], 0)
    bent = mg.line_plan([(0.0, 0.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0)], 41, dt)     # repeated point is dropped
    assert np.allclose(bent[20, :2], (0, 1)) and np.all((bent[:, 0] <= 1e-9) | (np.abs(bent[:, 1] - 1) <= 1e-9))
    return f'speed {np.linalg.norm(p[0, 2:]):.3f} per s'


def test_obstacles():
    cell = (5, 7)                                        # straight corridor running along axis 1
    assert ob.corridor_axis(MAZE, cell) == 1 and ob.corridor_axis(MAZE, (4, 6)) == 0
    assert ob.corridor_axis(MAZE, (3, 6)) is None and ob.corridor_axis(MAZE, (1, 1)) is None
    c = mu.cell_center(cell)
    through = np.stack([np.full(50, c[0]), np.linspace(c[1] - 1.5, c[1] + 1.5, 50)], axis=1)
    full = ob.Obstacle.full(cell, t_appear=10)
    assert ob.collides(through, full) and ob.ball_hits(c, full)
    assert not ob.collides(through[:10], full)
    assert ob.first_collision(through, full) == int(np.argmax(full.distance(through) < mu.BALL_RADIUS))
    # a plan that jumps over the cell between two points is still caught at the midpoint
    assert ob.collides(np.array([c + [0, -0.7], c + [0, 0.7]]), full)

    low = ob.Obstacle.partial(cell, travel_axis=1, side=-1, frac=0.5)
    assert np.allclose(low.rect, (4.3, 4.8, 6.3, 7.3)), low.rect
    g = low.gap_center()
    assert np.allclose(g, (5.05, 6.8)), g
    assert ob.collides(through, low)                     # the centre line grazes a half-width obstacle
    in_gap = through + [g[0] - c[0], 0.0]
    assert not ob.collides(in_gap, low) and ob.collides(in_gap, low, margin=0.2)
    assert not mu.point_in_wall(MAZE, g, mu.BALL_RADIUS)
    high = ob.Obstacle.partial(cell, travel_axis=1, side=+1, frac=0.5)
    assert np.allclose(high.gap_center(), (4.55, 6.8))
    assert np.allclose(full.gap_center(), c)
    assert ob.Obstacle.from_dict(low.to_dict()).rect == low.rect
    return 'full / partial / gap'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            print(f'[ OK ] {name}: {fn()}')
    print('ALL UNIT TESTS OK')
