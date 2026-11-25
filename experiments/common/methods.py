"""
The four planning loops compared in the project. Each adds one component (proposal §5):

  B0    plan once at t=0, never replan.
  B1    replan every `k_periodic` steps with best-of-N (obstacle-aware selection).
  B2    every `k` steps, check whether the REST of the plan hits a known obstacle; only then
        replan with best-of-N. No maze graph: if all N collide, follow the least-bad one.
  Ours  B2 + when all N collide: A* on the grid without the blocked cell -> waypoints ->
        subgoal chaining (one Diffuser plan per segment, horizon scaled to segment length).

All methods start with the same single plan (n=1, H=384), so they are identical until the
first replan. Parameters come from cfg['method_params'][<name>].
"""
import numpy as np

from experiments.common import maze_utils as mu
from experiments.common import waypoints as wpmod


class Method:
    name = 'base'

    def __init__(self, **params):
        self.p = params

    def reset(self, ctx):
        plans = ctx.sample(ctx.goal_xy, n=1, purpose='initial')
        ctx.adopt(plans[0], ctx.goal_xy, 'initial')

    def on_step(self, ctx):
        pass

    # shared helper ---------------------------------------------------------------------------
    def replan_horizon(self, ctx):
        """Horizon for a replan straight to the goal.
          None / 'full'  the trained 384. B0 README F1: Diffuser then arrives ~360 steps later
                         whatever the distance, so a replan after t ~ 440 cannot finish in 800 steps,
                         and periodic replanning keeps pushing the arrival back.
          'deadline'     (default) map-free: use the time that is left,
                         clip(round4(max_steps - t - deadline_slack), h_min, 384).
                         Early replans get the full 384 (room for a detour), late ones shrink so the
                         plan still ends before the step limit.
          'euclid'       map-free scaling with the straight-line distance to the goal:
                         clip(round4(spu * 1.5 * dist + margin), h_min, 384). Too short for a detour:
                         in the B2 probe almost no plan switched corridor with it.
          int            a fixed horizon."""
        mode = self.p.get('replan_horizon', 'deadline')
        if mode in (None, 'full'):
            return None
        if mode == 'deadline':
            left = ctx.cfg.get('max_steps', 800) - ctx.t - self.p.get('deadline_slack', 40)
            return int(min(ctx.H, max(self.p.get('h_min', 64), 4 * int(left // 4))))
        if mode == 'euclid':
            d = float(np.linalg.norm(ctx.goal_xy - ctx.obs[:2]))
            h = self.p.get('steps_per_cell', 30) * 1.5 * d + self.p.get('h_margin', 32)
            return int(min(ctx.H, max(self.p.get('h_min', 64), wpmod.round_up4(h))))
        return int(mode)

    def replan_to_goal(self, ctx, why):
        """Best-of-N straight to the goal. Returns True if a collision-free plan was found."""
        H = self.replan_horizon(ctx)
        plans = ctx.sample(ctx.goal_xy, n=self.p.get('n', 4), horizon=H, purpose=why)
        plan, all_collide, scores = ctx.best_of_n(plans)
        ctx.adopt(plan, ctx.goal_xy, why)
        ctx.event('best_of_n', n=len(scores), n_clear=sum(s['collision'] is None for s in scores),
                  n_ok=sum(s['ok'] for s in scores))
        if all_collide:
            ctx.counters['n_all_collide'] += 1
        return not all_collide


class B0(Method):
    name = 'B0'


class B1(Method):
    """Periodic replanning, whether or not anything is in the way."""
    name = 'B1'

    def on_step(self, ctx):
        k = self.p.get('k_periodic', 100)
        if ctx.t > 0 and ctx.t % k == 0:
            self.replan_to_goal(ctx, 'periodic')


class B2(Method):
    """Predictive trigger + best-of-N, no map."""
    name = 'B2'

    def check_due(self, ctx):
        k = self.p.get('k', 10)
        if ctx.t % k != 0 or not ctx.known_obstacles():
            return False
        ctx.counters['n_checks'] += 1
        if ctx.remaining_collides():
            ctx.counters['n_triggers'] += 1
            ctx.event('trigger')
            return True
        return False

    def on_step(self, ctx):
        if self.check_due(ctx):
            self.replan_to_goal(ctx, 'trigger')


class Ours(B2):
    """B2 + A* corridor switch + subgoal chaining through waypoints."""
    name = 'Ours'

    def reset(self, ctx):
        super().reset(ctx)
        self.chain = None           # list of remaining waypoint cells (last = goal cell)
        self.route = None           # A* route used for the chain
        self.inpaint_active = False # guidance == 'inpaint': rerouting with one pinned plan

    # -- chaining ------------------------------------------------------------------------------
    def _blocked_for_routing(self, ctx):
        # Cells to delete from the graph. Best-of-N already failed, so route around every known
        # obstacle cell, including partial ones.
        return {o.cell for o in ctx.known_obstacles()}

    def _start_chain(self, ctx):
        here = mu.xy_to_cell(ctx.obs[:2])
        blocked = self._blocked_for_routing(ctx)
        ctx.counters['n_astar'] += 1
        wps, route = wpmod.astar(ctx.maze, here, ctx.goal_cell, blocked - {here},
                                 mode=self.p.get('waypoint_mode', 'greedy'))
        if wps is None:
            ctx.counters['n_astar_fail'] += 1
            ctx.event('astar_no_route', here=list(here))
            return False
        ctx.counters['n_waypoints'] += len(wps) - 1
        ctx.event('astar', here=list(here), waypoints=[list(w) for w in wps], route_len=len(route) - 1)
        if self.p.get('guidance', 'chain') == 'inpaint':
            self.inpaint_active = True
            self._plan_inpaint(ctx, wps, route)
            return True
        self.chain, self.route = list(wps), route
        self._plan_segment(ctx, 'chain_start')
        return True

    def _plan_inpaint(self, ctx, wps, route):
        """Middle inpainting (proposal contribution 3): ONE plan from the ball to the goal whose
        x-y at step k_i is pinned to waypoint i. Horizon from the A* length (same rule as a chain
        segment); k_i proportional to the waypoint's distance along the route, scaled to where
        Diffuser actually arrives (about 24 steps before the plan end, B0 README F1)."""
        L = max(len(route) - 1, 1)
        H = wpmod.segment_horizon(L, self.p.get('steps_per_cell', 30), self.p.get('h_margin', 32),
                                  self.p.get('h_min', 64), ctx.H)
        arrive = H - self.p.get('arrive_slack', 24)
        via = []
        for w in wps[:-1]:
            d = route.index(tuple(w))
            k = int(min(H - 2, max(1, round(d / L * arrive))))
            via.append((k, mu.cell_center(w)))
        plans = ctx.sample(ctx.goal_xy, n=self.p.get('n', 4), horizon=H, purpose='inpaint', via=via)
        plan, failed, scores = ctx.best_of_n(plans)
        ctx.adopt(plan, ctx.goal_xy, 'inpaint')
        ctx.counters['n_segments'] += 1
        for _, xy in via:
            ctx.waypoints_xy.append(np.asarray(xy).copy())
        # plan quality around the pinned step: a "kink" is a step > jump_factor x the median step
        jf = self.p.get('jump_factor', 4.0)
        kinks = 0
        for p in plans:
            steps = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
            med = float(np.median(steps)) + 1e-6
            if any(steps[max(k - 2, 0):k + 2].max() > jf * med for k, _ in via):
                kinks += 1
        ctx.event('inpaint', H=H, ks=[k for k, _ in via], waypoints=[list(map(float, xy)) for _, xy in via],
                  n=len(scores), n_clear=sum(s['collision'] is None for s in scores),
                  n_wall_ok=sum(s['wall_depth'] <= ctx.max_wall_depth for s in scores),
                  n_ok=sum(s['ok'] for s in scores), n_kink=kinks)
        if failed:
            ctx.counters['n_all_collide'] += 1

    def _segment_goal(self, ctx):
        c = self.chain[0]
        return ctx.goal_xy if c == ctx.goal_cell else mu.cell_center(c)

    def _plan_segment(self, ctx, why):
        here = mu.xy_to_cell(ctx.obs[:2])
        target = self.chain[0]
        # grid length of this segment along the A* route (fallback: fresh shortest path)
        seg = wpmod.astar_path(ctx.maze, here, target, self._blocked_for_routing(ctx) - {here})
        n_cells = (len(seg) - 1) if seg else 1
        H = wpmod.segment_horizon(n_cells, self.p.get('steps_per_cell', 30), self.p.get('h_margin', 32),
                                  self.p.get('h_min', 64), ctx.H)
        goal_xy = self._segment_goal(ctx)
        plans = ctx.sample(goal_xy, n=self.p.get('n', 4), horizon=H, purpose=why)
        plan, all_collide, scores = ctx.best_of_n(plans)
        ctx.adopt(plan, goal_xy, why)
        ctx.counters['n_segments'] += 1
        ctx.waypoints_xy.append(goal_xy.copy())
        ctx.event('segment', target=list(target), H=H, n_cells=n_cells,
                  n_clear=sum(s['collision'] is None for s in scores), n_ok=sum(s['ok'] for s in scores))
        if all_collide:
            ctx.counters['n_all_collide'] += 1

    def on_step(self, ctx):
        # 1) chaining: move on to the next segment when the waypoint is reached or the plan is used up
        if self.chain and len(self.chain) > 1:
            wp_xy = mu.cell_center(self.chain[0])
            reached = np.linalg.norm(ctx.obs[:2] - wp_xy) <= self.p.get('waypoint_radius', 0.5)
            exhausted = ctx.plan_index() >= len(ctx.seq) - 1 + self.p.get('exhaust_slack', 20)
            if reached or exhausted:
                ctx.event('waypoint_reached' if reached else 'segment_exhausted', cell=list(self.chain[0]))
                self.chain.pop(0)
                self._plan_segment(ctx, 'next_segment')
                return
        # 2) predictive trigger (every k steps)
        if not self.check_due(ctx):
            return
        if self.inpaint_active:
            # already rerouting with a pinned plan: re-run A* from here and inpaint again
            self._start_chain(ctx)
            return
        if self.chain:
            # already rerouting: re-sample the current segment first, re-run A* if that fails too
            self._plan_segment(ctx, 'trigger_in_chain')
            if ctx.events[-1].get('n_ok') == 0:
                self._start_chain(ctx)
            return
        if not self.replan_to_goal(ctx, 'trigger'):
            self._start_chain(ctx)


class Oracle(Method):
    """Reference upper bound for the reaction (proposal §5: 已知地圖 A*＋PD 控制器).

    Until the obstacle appears it does exactly what every method does: follow the same first
    Diffuser plan (so it is in the same state when the obstacle appears; a map-based controller
    from t = 0 would move faster and be somewhere else). From the step the obstacle is known it
    drops the Diffuser: A* over the maze cells without the obstacle cell, followed with D4RL's
    own maze2d waypoint PD controller
        action = clip(p_gain * (target - pos) + d_gain * vel, -1, 1),  p_gain = 10, d_gain = -1,
    switching to the next cell centre within `switch_radius`; the last target is the goal.
    It never samples another plan."""
    name = 'Oracle'

    def reset(self, ctx):
        super().reset(ctx)
        self.known = set()
        self.targets = None

    def _route(self, ctx, blocked):
        here = mu.xy_to_cell(ctx.obs[:2])
        path = wpmod.astar_path(ctx.maze, here, ctx.goal_cell, set(blocked) - {here})
        ctx.counters['n_astar'] += 1
        if path is None:
            ctx.counters['n_astar_fail'] += 1
            ctx.event('astar_no_route', here=list(here))
            path = [here, ctx.goal_cell]
        pts = [mu.cell_center(c) for c in path[1:-1]] + [ctx.goal_xy]
        self.targets = [np.asarray(p, dtype=np.float64) for p in pts]
        seq = np.vstack([ctx.obs[:2][None]] + [p[None] for p in self.targets])
        ctx.adopt(np.hstack([seq, np.zeros_like(seq)]), ctx.goal_xy, 'oracle_route')
        ctx.event('astar', here=list(here), route_len=len(path) - 1)

    def on_step(self, ctx):
        cells = {o.cell for o in ctx.known_obstacles()}
        if cells != self.known:
            self.known = cells
            self._route(ctx, cells)

    def action(self, ctx, state):
        if self.targets is None:                 # before the obstacle: same as everyone
            from experiments.common.rollout import waypoint_action
            return waypoint_action(ctx.seq, ctx.plan_index(), state)
        r = self.p.get('switch_radius', 0.25)
        while len(self.targets) > 1 and np.linalg.norm(state[:2] - self.targets[0]) < r:
            self.targets.pop(0)
        a = self.p.get('p_gain', 10.0) * (self.targets[0] - state[:2]) + self.p.get('d_gain', -1.0) * state[2:4]
        return np.clip(a, -1.0, 1.0)


METHODS = {'B0': B0, 'B1': B1, 'B2': B2, 'Ours': Ours, 'Oracle': Oracle}


def make_method(name, cfg):
    params = dict(cfg.get('method_params', {}).get('common', {}))
    params.update(cfg.get('method_params', {}).get(name, {}))
    return METHODS[name](**params)
