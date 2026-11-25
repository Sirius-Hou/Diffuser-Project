"""
Corridor choice on the maze grid graph (numpy only, no mujoco).

Team interface:  astar(start_cell, goal_cell, blocked_cells) -> waypoint list
  = astar_path(...)  then  select_waypoints(...)

Waypoint modes
  'greedy'     (default) the fewest waypoints that force the new corridor: starting from the
               current anchor, keep the farthest cell on the A* route such that NO shortest path
               of the obstacle-unaware maze from the anchor to that cell passes a blocked cell.
               Rationale: the frozen Diffuser imitates near-shortest routes of the unblocked maze,
               so a segment is only safe if every such route avoids the blockage.
  'junctions'  every junction cell (>= 3 free neighbours) on the route (proposal wording:
               "路口格當 waypoint"). More segments, each a single corridor.
The goal cell is always the last waypoint.
"""
import heapq
from collections import deque

import numpy as np

from experiments.common import maze_utils as mu


def astar_path(maze_arr, start, goal, blocked=()):
    """A* (Hart et al. 1968) on the 4-connected free cells, Manhattan heuristic.
    Returns the list of cells start..goal (inclusive), or None if unreachable."""
    blocked = set(map(tuple, blocked))
    start, goal = tuple(start), tuple(goal)
    if start in blocked or goal in blocked:
        return None
    h = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])  # noqa: E731
    g = {start: 0}
    prev = {start: None}
    heap = [(h(start), 0, start)]
    while heap:
        _, gc, c = heapq.heappop(heap)
        if c == goal:
            path = []
            while c is not None:
                path.append(c)
                c = prev[c]
            return path[::-1]
        if gc > g[c]:
            continue
        for d in mu.NEIGHBORS4:
            n = (c[0] + d[0], c[1] + d[1])
            if not mu.is_free(maze_arr, n) or n in blocked:
                continue
            ng = gc + 1
            if ng < g.get(n, 1 << 30):
                g[n] = ng
                prev[n] = c
                heapq.heappush(heap, (ng + h(n), ng, n))
    return None


def bfs_dist(maze_arr, src, blocked=()):
    """Grid distance from src to every reachable free cell."""
    blocked = set(map(tuple, blocked))
    dist = {tuple(src): 0}
    q = deque([tuple(src)])
    while q:
        c = q.popleft()
        for d in mu.NEIGHBORS4:
            n = (c[0] + d[0], c[1] + d[1])
            if mu.is_free(maze_arr, n) and n not in blocked and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def junction_cells(maze_arr):
    return {(int(c[0]), int(c[1])) for c in mu.free_cells(maze_arr)
            if sum(mu.is_free(maze_arr, (c[0] + d[0], c[1] + d[1])) for d in mu.NEIGHBORS4) >= 3}


def _blockage_on_some_shortest_path(maze_arr, a, c, blocked, dist_cache):
    """True if some shortest path a->c in the UNBLOCKED maze passes a blocked cell."""
    da = dist_cache.setdefault(a, bfs_dist(maze_arr, a))
    dc = dist_cache.setdefault(c, bfs_dist(maze_arr, c))
    d = da.get(c)
    if d is None:
        return True
    return any(b in da and b in dc and da[b] + dc[b] == d for b in blocked)


def select_waypoints(maze_arr, path, blocked, mode='greedy'):
    """Waypoint cells along an A* route `path` (list of cells, path[0] = current cell).
    Always ends with path[-1] (the goal cell)."""
    path = [tuple(c) for c in path]
    blocked = set(map(tuple, blocked))
    if len(path) <= 1:
        return list(path[-1:])
    if mode == 'junctions':
        J = junction_cells(maze_arr)
        wps = [c for c in path[1:-1] if c in J]
        return wps + [path[-1]]
    if mode != 'greedy':
        raise ValueError(mode)
    cache, wps, a_idx = {}, [], 0
    while a_idx < len(path) - 1:
        a = path[a_idx]
        nxt = a_idx + 1                       # always progress at least one cell
        for k in range(len(path) - 1, a_idx, -1):
            if not _blockage_on_some_shortest_path(maze_arr, a, path[k], blocked, cache):
                nxt = k
                break
        wps.append(path[nxt])
        a_idx = nxt
    return wps


def astar(maze_arr, start_cell, goal_cell, blocked, mode='greedy'):
    """Team interface. Returns (waypoint cells, full A* route) or (None, None) if no route."""
    path = astar_path(maze_arr, start_cell, goal_cell, blocked)
    if path is None:
        return None, None
    return select_waypoints(maze_arr, path, blocked, mode), path


def round_up4(x):
    return int(4 * int(np.ceil(x / 4.0)))


def segment_horizon(n_cells, steps_per_cell=30.0, margin=32, h_min=64, h_max=384):
    """Plan length for a segment that covers `n_cells` grid steps.

    B0 (README F1): Diffuser stretches every plan over its whole horizon (arrives ~24 steps
    before the end), so a segment's horizon sets how long the ball takes. Scale it with the
    segment length instead of using the full 384. Must be a multiple of 4 (U-Net down/up-sampling).
    """
    h = steps_per_cell * max(n_cells, 1) + margin
    return int(min(h_max, max(h_min, round_up4(h))))
