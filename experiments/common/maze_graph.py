"""
Grid graph of the maze: A* route around blocked cells and the waypoints on that route
(shared by our method, the oracle and the scenario filter).

Team interface 3:  astar(maze, start_cell, goal_cell, blocked) -> list of cells
                   waypoints(maze, path) -> indices into the path
"""
import heapq

from . import maze_utils as mu


def degree(maze_arr, cell, blocked=()):
    """Number of free, unblocked 4-neighbours."""
    return sum(mu.is_free(maze_arr, (cell[0] + d[0], cell[1] + d[1])) and
               (cell[0] + d[0], cell[1] + d[1]) not in blocked for d in mu.NEIGHBORS4)


def astar(maze_arr, start, goal, blocked=()):
    """Shortest 4-connected path of cells from start to goal (inclusive) avoiding `blocked`, or None.
    Unit edge costs with the Manhattan heuristic, so the length equals maze_utils.bfs_path."""
    start, goal = tuple(start), tuple(goal)
    blocked = set(map(tuple, blocked))
    if goal in blocked or not mu.is_free(maze_arr, goal):
        return None

    def h(c):
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

    g = {start: 0}
    prev = {start: None}
    heap = [(h(start), 0, start)]
    closed = set()
    while heap:
        _, gc, c = heapq.heappop(heap)
        if c in closed:
            continue
        if c == goal:
            break
        closed.add(c)
        for d in mu.NEIGHBORS4:
            n = (c[0] + d[0], c[1] + d[1])
            if not mu.is_free(maze_arr, n) or n in blocked or n in closed:
                continue
            if gc + 1 < g.get(n, float('inf')):
                g[n] = gc + 1
                prev[n] = c
                heapq.heappush(heap, (gc + 1 + h(n), gc + 1, n))
    if goal not in prev:
        return None
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev[c]
    return path[::-1]


def is_corner(path, i):
    """True if the path turns at index i."""
    a, b, c = path[i - 1], path[i], path[i + 1]
    return (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])


def waypoints(maze_arr, path, blocked=(), max_leg=None, corners=False):
    """Indices into `path` of the waypoints: every junction cell after the start, then the goal.

    A junction is a cell with 3 or more open neighbours: the places where a corridor is chosen.
    corners=True also makes every cell where the path turns a waypoint, so each leg is a straight
    line (r3_horizon: the frozen Diffuser cuts through walls on legs that bend back on themselves).
    If max_leg is set, a leg longer than max_leg cells is split at its last corner within reach
    (or at max_leg cells if the leg has no corner there), so every leg fits one short plan."""
    blocked = set(map(tuple, blocked))
    last = len(path) - 1
    idx = [i for i in range(1, last)
           if degree(maze_arr, path[i], blocked) >= 3 or (corners and is_corner(path, i))]
    idx.append(last)
    if not max_leg:
        return idx
    out, prev = [], 0
    for i in idx:
        while i - prev > max_leg:
            reach = prev + max_leg
            turns = [j for j in range(prev + 1, reach + 1) if is_corner(path, j)]
            prev = turns[-1] if turns else reach
            out.append(prev)
        out.append(i)
        prev = i
    return out


def leg_lengths(wp_idx):
    """Cells travelled in each leg, given waypoint indices into a path that starts at index 0."""
    return [b - a for a, b in zip([0] + list(wp_idx[:-1]), wp_idx)]
