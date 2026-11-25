"""
Shared episode loop for every method (B0, B1, B2, ours) on blocked-corridor scenarios.

The loop owns everything that must be identical across methods:
  seeding and reset order (same as B0, so every method's FIRST plan is bit-identical),
  when the obstacle appears, the ball-vs-obstacle check, the controller, success, metrics.
A method only decides WHEN to replan and WHICH plan to follow (see methods.py).

Works with the real env + FrozenDiffuserPlanner, or with fakes.FakePointEnv + fakes.FakePlanner
(for testing the logic on a machine without mujoco).
"""
import random
import time

import numpy as np

from experiments.common import maze_utils as mu
from experiments.common import obstacles as obsmod


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def waypoint_action(sequence, i, state):
    """The original plan_maze2d.py controller, with i = steps since this plan was made."""
    if i < len(sequence) - 1:
        nxt = sequence[i + 1]
    else:
        nxt = sequence[-1].copy()
        nxt[2:] = 0
    return nxt[:2] - state[:2] + (nxt[2:] - state[2:])


def densify(xys, step=0.05):
    """Insert points so consecutive points are <= step apart (for collision checks)."""
    xys = np.asarray(xys, dtype=np.float64)
    if len(xys) < 2:
        return xys
    out = [xys[:1]]
    for a, b in zip(xys[:-1], xys[1:]):
        n = int(np.ceil(np.linalg.norm(b - a) / step))
        if n > 1:
            out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
        else:
            out.append(b[None])
    return np.concatenate(out)


def wall_depth(maze_arr, xy):
    """How deep a point lies inside a wall: 0 in a free cell, else the distance to the nearest
    free cell (corner-cutting gives a few hundredths; tunnelling through a wall gives ~0.5)."""
    c = mu.xy_to_cell(xy)
    if mu.is_free(maze_arr, c):
        return 0.0
    best = 1.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            n = (c[0] + di, c[1] + dj)
            if not mu.is_free(maze_arr, n):
                continue
            dx = max(n[0] + mu.CELL_LO - xy[0], 0.0, xy[0] - (n[0] + mu.CELL_HI))
            dy = max(n[1] + mu.CELL_LO - xy[1], 0.0, xy[1] - (n[1] + mu.CELL_HI))
            best = min(best, float(np.hypot(dx, dy)))
    return best


def max_wall_depth(maze_arr, xys):
    return max((wall_depth(maze_arr, p) for p in np.asarray(xys)[:, :2]), default=0.0)


class EpisodeContext:
    """State shared between the loop and a method during one episode."""

    def __init__(self, env, planner, scenario, cfg, maze_arr):
        self.env, self.planner, self.cfg = env, planner, cfg
        self.maze = maze_arr
        self.scenario = scenario
        self.obstacles = [obsmod.Obstacle.from_dict(o) for o in scenario.get('obstacles', [])]
        self.goal_xy = np.array(env._target, dtype=np.float64)
        self.goal_cell = mu.xy_to_cell(self.goal_xy)
        self.H = planner.horizon
        self.inflate = mu.BALL_RADIUS + cfg.get('collision_margin', 0.05)
        self.max_wall_depth = cfg.get('max_wall_depth', 0.2)
        # current plan
        self.seq = None
        self.plan_t0 = 0
        self.plan_goal = None
        # bookkeeping
        self.t = 0
        self.obs = None
        self.plan_calls = []          # dicts: t, n, H, seconds, purpose
        self.plans_followed = []      # (t, seq) every plan that was adopted
        self.events = []              # (t, name, info) timeline for failure analysis
        self.counters = dict(n_checks=0, n_triggers=0, n_all_collide=0, n_astar=0,
                             n_astar_fail=0, n_waypoints=0, n_segments=0, n_replans=0)
        self.waypoints_xy = []        # every waypoint used (for plots)

    # -- what the method can see --------------------------------------------------------------
    def known_obstacles(self):
        """Perfect sensing: an obstacle is known from the step it appears."""
        return [o for o in self.obstacles if o.active(self.t)]

    def plan_index(self):
        return self.t - self.plan_t0

    def remaining_plan_xy(self):
        """Where the ball is headed from now on: current position, then the rest of the plan
        (or the straight line to the plan's end once the plan is used up)."""
        i = min(self.plan_index() + 1, len(self.seq) - 1)
        pts = np.vstack([self.obs[:2][None], self.seq[i:, :2]])
        return densify(pts)

    def check_inflate(self):
        """Safety margin for plan checks. If the ball is already inside the margin of an obstacle
        (but not touching it), every plan starting at the ball would count as colliding at its
        first point; then use the bare ball radius so plans that move away are not rejected."""
        known = self.known_obstacles()
        if any(obsmod.disc_hits_rect(self.obs[:2], o.rect(), self.inflate) for o in known):
            return mu.BALL_RADIUS
        return self.inflate

    def remaining_collides(self):
        known = self.known_obstacles()
        return bool(known) and obsmod.collides(self.remaining_plan_xy(), known, inflate=self.check_inflate())

    # -- planning ------------------------------------------------------------------------------
    def sample(self, goal_xy, n, horizon=None, purpose='', via=None):
        kw = dict(via=via) if via else {}
        plans, _, dt = self.planner.plan(self.obs, goal_xy, n=n, horizon=horizon, **kw)
        self.plan_calls.append(dict(t=self.t, n=int(n), H=int(horizon or self.H), seconds=float(dt),
                                    purpose=purpose))
        return np.asarray(plans)

    def score_plans(self, plans):
        """Per plan: first obstacle-collision index (None = clear), wall points, deepest point
        inside a wall, length, and ok = clear and not tunnelling through a wall."""
        known = self.known_obstacles()
        infl = self.check_inflate()
        out = []
        for k, p in enumerate(plans):
            xy = densify(p[:, :2])
            coll = obsmod.first_collision(xy, known, inflate=infl) if known else None
            depth = max_wall_depth(self.maze, p[:, :2])
            out.append(dict(k=k, collision=coll, wall_pts=mu.count_points_in_wall(self.maze, p[:, :2]),
                            wall_depth=depth, length=mu.path_length(p[:, :2]),
                            ok=coll is None and depth <= self.max_wall_depth))
        return out

    def best_of_n(self, plans):
        """best-of-N. A plan is OK if it avoids every known obstacle AND does not tunnel through a
        wall (deepest point <= max_wall_depth; corner-cutting is allowed). Among OK plans take the
        shortest. Returns (plan, failed: bool, scores); failed = no OK plan, in which case the
        least-bad plan is returned: clear of the obstacle with the shallowest wall crossing,
        otherwise the one whose first collision is latest."""
        scores = self.score_plans(plans)
        ok = [s for s in scores if s['ok']]
        if ok:
            best = min(ok, key=lambda s: s['length'])
            return plans[best['k']], False, scores
        clear = [s for s in scores if s['collision'] is None]
        if clear:
            best = min(clear, key=lambda s: s['wall_depth'])
        else:
            best = max(scores, key=lambda s: s['collision'])
        return plans[best['k']], True, scores

    def adopt(self, seq, goal_xy, why):
        if self.seq is not None:
            self.counters['n_replans'] += 1
        self.seq = np.asarray(seq, dtype=np.float64)
        self.plan_t0 = self.t
        self.plan_goal = np.asarray(goal_xy, dtype=np.float64)
        self.plans_followed.append((self.t, self.seq.copy()))
        self.event('adopt_plan', why=why, H=len(self.seq))

    def event(self, name, **info):
        self.events.append(dict(t=int(self.t), event=name, **info))


def start_episode(env, seed):
    """Seeding and reset in exactly the B0 order."""
    set_all_seeds(seed)
    env.seed(seed)
    return env.reset()


def run_episode(env, planner, method, scenario, cfg, maze_arr, contact_fn=None, stop_at_t=None):
    """Run one episode. Returns (record dict, trajectory arrays dict, ctx).

    stop_at_t  if set, stop right before acting at that step (used to replay B0 up to a time).
    """
    contact_fn = contact_fn or (lambda e: mu.wall_contact(e))
    success_radius = cfg.get('success_radius', 0.5)
    max_steps = cfg.get('max_steps', env.max_episode_steps)
    seed = scenario['seed']

    t_ep0 = time.perf_counter()
    ctx = EpisodeContext(env, planner, scenario, cfg, maze_arr)
    ctx.obs = start_episode(env, seed)
    start_xy = ctx.obs[:2].copy()
    ctx.t = 0
    method.reset(ctx)                       # must adopt the initial plan

    rollout_xy = [ctx.obs[:2].copy()]
    contact = []
    first_success_t, collision_t = None, None
    stop_reason = 'max_steps'
    for t in range(max_steps):
        ctx.t = t
        if stop_at_t is not None and t >= stop_at_t:
            stop_reason = 'stop_at_t'
            break
        for ob in ctx.obstacles:
            if ob.appear_t == t:
                ctx.event('obstacle_appears', cell=list(ob.cell), kind=ob.kind)
        method.on_step(ctx)
        state = env.state_vector().copy()
        if hasattr(method, 'action'):           # methods with their own controller (Oracle)
            action = method.action(ctx, state)
        else:
            action = waypoint_action(ctx.seq, ctx.plan_index(), state)
        obs, _, terminal, _ = env.step(action)
        ctx.obs = obs
        xy = obs[:2].copy()
        rollout_xy.append(xy)
        contact.append(bool(contact_fn(env)))
        if obsmod.ball_collides(xy, ctx.obstacles, t + 1):
            collision_t = t + 1
            ctx.t = t + 1
            ctx.event('obstacle_collision')
            stop_reason = 'obstacle_collision'
            break
        if np.linalg.norm(xy - ctx.goal_xy) <= success_radius:
            first_success_t = t + 1
            stop_reason = 'success'
            break
        if terminal:
            stop_reason = 'terminal'
            break
    episode_time = time.perf_counter() - t_ep0

    rollout_xy = np.array(rollout_xy)
    contact = np.array(contact, dtype=bool)
    dists = np.linalg.norm(rollout_xy - ctx.goal_xy, axis=1)
    success = first_success_t is not None and collision_t is None
    stalled = len(rollout_xy) > 100 and mu.path_length(rollout_xy[-101:]) < 0.1
    late_contact = bool(contact[-100:].any()) if len(contact) else False
    if success:
        failure = ''
    elif collision_t is not None:
        failure = 'obstacle_collision'
    elif stop_reason == 'stop_at_t':
        failure = 'stopped'
    elif stalled and late_contact:
        failure = 'stuck_on_wall'
    elif stalled:
        failure = 'stalled_off_goal'
    else:
        failure = 'timeout_still_moving'

    appear = min([o.appear_t for o in ctx.obstacles], default=None)
    react = [e['t'] for e in ctx.events if e['event'] == 'adopt_plan' and appear is not None
             and e['t'] >= appear]
    pt = [c['seconds'] for c in ctx.plan_calls]
    start_cell = mu.xy_to_cell(start_xy)
    rec = dict(
        method=method.name, seed=int(seed), scenario_id=scenario.get('id', f'seed{seed}'),
        obstacles=[o.to_dict() for o in ctx.obstacles], appear_t=appear,
        start_cell=f'{start_cell[0]},{start_cell[1]}',
        # outcome
        success=bool(success), first_success_step=first_success_t,
        obstacle_collision=collision_t is not None, collision_step=collision_t,
        final_dist_to_goal=float(dists[-1]), min_dist_to_goal=float(dists.min()),
        steps=len(rollout_xy) - 1, stop_reason=stop_reason, failure_reason=failure,
        # cost
        planning_time_s=float(sum(pt)), n_plan_calls=len(pt),
        n_plans=int(sum(c['n'] for c in ctx.plan_calls)),
        planning_time_per_call_max_s=float(max(pt)) if pt else 0.0,
        episode_time_s=float(episode_time),
        # replanning behaviour
        first_replan_after_obstacle=(react[0] - appear) if react else None,
        **ctx.counters,
        # path / contacts
        path_length=mu.path_length(rollout_xy),
        wall_contact_steps=int(contact.sum()), wall_contact_events=mu.contact_events(contact),
        visited_blocked_cell=any(mu.xy_to_cell(p) == o.cell for p in rollout_xy for o in ctx.obstacles),
    )
    plan_ts = np.array([t for t, _ in ctx.plans_followed])
    plan_lens = np.array([len(s) for _, s in ctx.plans_followed])
    traj = dict(rollout_xy=rollout_xy, contact=contact, plan_ts=plan_ts, plan_lens=plan_lens,
                plans_concat=np.concatenate([s for _, s in ctx.plans_followed]),
                waypoints_xy=np.array(ctx.waypoints_xy).reshape(-1, 2))
    return rec, traj, ctx


def render_episode(path, maze_arr, rec, traj, ctx, goal_xy, success_radius=0.5):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6, 4.8))
    mu.draw_maze(ax, maze_arr)
    obsmod.draw_obstacles(ax, ctx.obstacles)
    plans = np.split(traj['plans_concat'], np.cumsum(traj['plan_lens'])[:-1])
    for k, (t0, p) in enumerate(zip(traj['plan_ts'], plans)):
        lab = 'plan t=0' if k == 0 else ('replans' if k == 1 else None)
        ax.plot(p[:, 1], p[:, 0], color='tab:blue' if k == 0 else 'tab:cyan', lw=1.0 if k == 0 else 0.7,
                alpha=0.8 if k == 0 else 0.6, label=lab)
    r = traj['rollout_xy']
    ax.plot(r[:, 1], r[:, 0], color='tab:orange', lw=1.6, label='executed')
    if len(traj['waypoints_xy']):
        w = traj['waypoints_xy']
        ax.scatter(w[:, 1], w[:, 0], marker='*', s=90, color='gold', edgecolor='k', zorder=5, label='waypoints')
    ax.scatter([r[0, 1]], [r[0, 0]], color='green', s=40, zorder=5, label='start')
    if rec['obstacle_collision']:
        ax.scatter([r[-1, 1]], [r[-1, 0]], marker='x', color='red', s=80, zorder=6, label='collision')
    ax.add_patch(Circle((goal_xy[1], goal_xy[0]), success_radius, fill=False, color='magenta', lw=1.5))
    status = 'SUCCESS' if rec['success'] else f"FAIL ({rec['failure_reason']})"
    ax.set_title(f"{rec['method']} {rec['scenario_id']}: {status}\n"
                 f"obstacle t={rec['appear_t']}, replans={rec['n_replans']}, plans={rec['n_plans']}, "
                 f"plan time {rec['planning_time_s']:.1f}s", fontsize=8)
    ax.legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
