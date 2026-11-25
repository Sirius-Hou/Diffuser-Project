"""
Tests for the numpy-only shared code (runs anywhere, no mujoco / torch / weights):

    python experiments/common/test_common.py

Also runs one fake episode per method to check the loop wiring.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common import waypoints as wp  # noqa: E402
from experiments.common.mazes import load_maze_arr  # noqa: E402
from experiments.common.rollout import densify  # noqa: E402

M = load_maze_arr('maze2d-large-v1')


def test_maze_layout():
    assert M.shape == (9, 12)
    assert mu.xy_to_cell((7.0, 9.0)) == (7, 9) and M[7, 9] == mu.GOAL
    assert M[6, 9] == mu.WALL            # the wall B0 failures get stuck on (B0 README F2)
    assert len(mu.free_cells(M)) == 46   # free cells incl. the goal


def test_astar_matches_bfs():
    free = [tuple(map(int, c)) for c in mu.free_cells(M)]
    for a in free[::3]:
        for b in free[::4]:
            p, q = wp.astar_path(M, a, b), mu.bfs_path(M, a, b)
            assert len(p) == len(q), (a, b)
            assert all(abs(p[i][0] - p[i + 1][0]) + abs(p[i][1] - p[i + 1][1]) == 1 for i in range(len(p) - 1))


def test_blocked_route_and_waypoints():
    start, goal = (3, 1), (7, 9)
    assert wp.astar_path(M, start, goal, {(3, 5)}) is None          # (3,5) is a bridge: no detour
    wps, route = wp.astar(M, start, goal, {(4, 6)})
    assert route is not None and (4, 6) not in route
    assert wps[-1] == goal
    # every segment between consecutive anchors must avoid the blockage on ALL shortest paths
    anchors = [start] + wps
    for a, b in zip(anchors[:-1], anchors[1:]):
        assert not wp._blockage_on_some_shortest_path(M, a, b, {(4, 6)}, {})
    jw = wp.select_waypoints(M, route, {(4, 6)}, mode='junctions')
    assert jw[-1] == goal and all(c in wp.junction_cells(M) for c in jw[:-1])


def test_obstacle_geometry():
    full = ob.Obstacle(cell=(4, 6), appear_t=100)
    (x0, x1), (y0, y1) = full.rect()
    assert np.allclose([x0, x1, y0, y1], [3.3, 4.3, 5.3, 6.3])
    c = mu.cell_center((4, 6))
    assert ob.disc_hits_rect(c, full.rect(), mu.BALL_RADIUS)
    assert not ob.disc_hits_rect(mu.cell_center((3, 6)), full.rect(), mu.BALL_RADIUS)
    assert ob.ball_collides(c, [full], t=100) and not ob.ball_collides(c, [full], t=99)
    # corridor (4,6) runs along x (free above and below) -> cross axis is y
    assert ob.corridor_axis(M, (4, 6)) == 1 and ob.corridor_axis(M, (3, 6)) is None
    half = ob.Obstacle(cell=(4, 6), appear_t=0, kind='partial', frac=0.5, side='lo', axis=1)
    assert not half.blocks_cell
    (_, _), (py0, py1) = half.rect()
    assert np.isclose(py0, 5.3) and np.isclose(py1, 5.8)
    assert not ob.disc_hits_rect(np.array([c[0], 6.05]), half.rect(), mu.BALL_RADIUS)   # through the gap
    assert ob.Obstacle(cell=(4, 6), appear_t=0, kind='partial', frac=0.85, axis=1).blocks_cell
    assert ob.Obstacle.from_dict(full.to_dict()) == full


def test_plan_collision_check():
    full = ob.Obstacle(cell=(4, 6), appear_t=0)
    down = densify(np.array([mu.cell_center((3, 6)), mu.cell_center((5, 6))]))
    assert ob.first_collision(down, [full]) is not None
    side = densify(np.array([mu.cell_center((3, 4)), mu.cell_center((3, 6))]))
    assert ob.first_collision(side, [full]) is None
    assert ob.first_collision(down, [ob.Obstacle(cell=(4, 6), appear_t=50)], t=10) is None


def test_wall_depth():
    from experiments.common.rollout import wall_depth
    assert wall_depth(M, mu.cell_center((3, 1))) == 0.0
    assert np.isclose(wall_depth(M, mu.cell_center((4, 2))), 0.5)          # middle of a 1-cell wall
    assert wall_depth(M, np.array([3.35, 1.8])) < 0.1                      # just past a wall edge


def test_segment_horizon():
    for n in range(0, 30):
        h = wp.segment_horizon(n)
        assert h % 4 == 0 and 64 <= h <= 384


def test_fake_episodes():
    from experiments.common import fakes
    from experiments.common.methods import make_method
    from experiments.common.rollout import run_episode
    env = fakes.FakePointEnv()
    planner = fakes.FakePlanner(env.maze_arr)
    cfg = dict(method_params=dict(common=dict(n=4, replan_horizon='euclid')))
    scen = dict(id='t', seed=20, obstacles=[ob.Obstacle(cell=(5, 7), appear_t=61).to_dict()])
    out = {}
    for m in ['B0', 'B1', 'B2', 'Ours', 'Oracle']:
        rec, traj, ctx = run_episode(env, planner, make_method(m, cfg), scen, cfg, env.maze_arr, fakes.fake_contact)
        out[m] = rec
        # determinism: same seed -> same rollout
        rec2, traj2, _ = run_episode(env, planner, make_method(m, cfg), scen, cfg, env.maze_arr, fakes.fake_contact)
        assert np.array_equal(traj['rollout_xy'], traj2['rollout_xy']), m
    assert out['B0']['obstacle_collision'] and out['B0']['n_plans'] == 1
    assert out['Ours']['n_astar'] >= 1 and out['Ours']['success']
    inp = dict(cfg, method_params=dict(common=dict(n=4, replan_horizon='euclid'), Ours=dict(guidance='inpaint')))
    rec, _, ctx = run_episode(env, planner, make_method('Ours', inp), scen, inp, env.maze_arr, fakes.fake_contact)
    ev = [e for e in ctx.events if e['event'] == 'inpaint']
    assert ev and all(0 < k < ev[0]['H'] - 1 for k in ev[0]['ks']), ev
    assert out['Oracle']['success'] and out['Oracle']['n_plans'] == 1 and out['Oracle']['n_astar'] == 1


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok  ', t.__name__)
    print(f'ALL {len(tests)} TESTS PASSED')
