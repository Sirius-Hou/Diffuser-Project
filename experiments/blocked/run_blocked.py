"""
Blocked-corridor evaluation: one planning loop per method around the same frozen Diffuser.

Every method starts from the stored state and t=0 plan of each scenario (see make_scenarios.py).
The obstacle becomes known at t_appear; touching it ends the episode as a failure.

  B0     : never replans
  B1     : replans every `k_fixed` steps, best-of-N against the obstacle once it is known
  B2     : every `k` steps checks the rest of the plan against the obstacle; replans (best-of-N) only on a hit
  ours   : B2, and when all N samples hit: A* around the blocked cell, then one short plan per waypoint.
           With gap_route, a partial obstacle that no route avoids is passed through its gap on waypoint legs
  astar_line : ours with every waypoint leg drawn as a constant-speed line instead of sampled from the
           Diffuser (ablation: is the Diffuser needed once A* has chosen the corridor?)
  oracle : A* on the known map with a PD controller (reference only)

    bash experiments/common/setup/run.sh python experiments/blocked/run_blocked.py --config experiments/blocked/configs/smoke.json --method ours
Options: --seeds 3 4   --device cpu   --no_png
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_graph as mg  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds, waypoint_action  # noqa: E402
from experiments.b0_baseline.run_b0 import git_commit  # noqa: E402

METHODS = ('B0', 'B1', 'B2', 'ours', 'astar_line', 'oracle')


class PlanningLoop:
    """Decides, before every control step, whether to replace the current plan."""

    def __init__(self, method, env, planner, cfg, obstacle, plan0, target):
        self.method, self.env, self.planner, self.cfg = method, env, planner, cfg
        self.reroutes = method in ('ours', 'astar_line')    # falls back to A* when every sample hits
        self.maze, self.obstacle, self.target = env.maze_arr, obstacle, target
        self.goal_cell = mu.xy_to_cell(target)
        self.seq, self.t0 = plan0, 0                # current plan and the step it started at
        self.legs, self.leg_i = None, 0             # ours: waypoint legs once A* has been used
        self.route_blocked = [obstacle.cell]        # ours: cells the legs' A* avoids ([] through a gap)
        self.path_xy = None                         # oracle: route as a list of points
        self.plans = [] if method == 'oracle' else [(0, plan0)]     # every adopted plan, for plotting
        self.events = []
        self.planning_time = 0.0
        self.n_calls = self.n_samples = self.n_triggers = self.n_astar = 0

    def log(self, t, event, **info):
        self.events.append(dict(t=int(t), event=event, **info))

    # ---- sampling and selection ---------------------------------------------------------------

    def goal_horizon(self, t):
        """Horizon for a plan to the goal.
          'full'      : always the trained horizon
          'remaining' : shortened so that the plan ends before the episode's step limit
          'shrink'    : shortened so that the plan ends when the t=0 plan would have ended"""
        H, m = self.planner.horizon, self.planner.horizon_multiple
        mode = self.cfg['replan_horizon']
        if mode == 'full':
            return H
        left = (H - t) if mode == 'shrink' else (self.env.max_episode_steps - t - self.cfg['budget_slack_steps'])
        return int(min(H, max(self.cfg['min_horizon'], m * (left // m))))

    def sample(self, t, state, goal_xy, n, horizon, use_obstacle, route=None):
        """Sample n plans and adopt the first one clear of the obstacle (else the one that hits it latest).

        route (ours, waypoint legs): cells of the A* route for this leg. Among the clear samples, prefer the
        one that visits the fewest cells off this route, then the fewest points inside a wall, then the one
        that ends nearest to where it was told to end (short plans can stop short and jump to the
        inpainted last point)."""
        plans, _, dt = self.planner.plan(state, goal_xy, n=n, horizon=horizon)
        self.planning_time += dt
        self.n_calls += 1
        self.n_samples += n
        if use_obstacle:
            hits = [ob.first_collision(p[:, :2], self.obstacle, self.cfg['check_margin']) for p in plans]
            clear = [i for i, h in enumerate(hits) if h is None]
        else:
            hits, clear = [None] * n, list(range(n))
        n_on_route = None
        if not clear:
            best = int(np.argmax(hits))
        elif route is None:
            best = clear[0]
        else:
            route = set(route)
            keys = {}
            for i in clear:
                xy = plans[i][:, :2]
                keys[i] = (len(set(mu.cell_sequence(xy)) - route), mu.count_points_in_wall(self.maze, xy),
                           float(np.linalg.norm(xy[-1] - xy[-2])))
            best = min(clear, key=keys.get)
            n_on_route = sum(k[0] == 0 for k in keys.values())
        self.seq, self.t0 = plans[best], t
        self.plans.append((t, plans[best]))
        return len(clear), n_on_route

    # ---- ours: A* route and waypoint legs ------------------------------------------------------

    def needs_gap(self, state):
        """gap_route: the obstacle is partial and every route to the goal passes through its cell."""
        return (self.cfg.get('gap_route', False) and self.obstacle.kind == 'partial' and
                mg.astar(self.maze, mu.xy_to_cell(state[:2]), self.goal_cell, [self.obstacle.cell]) is None)

    def start_route(self, t, state):
        cur = mu.xy_to_cell(state[:2])
        gap = self.needs_gap(state)
        blocked = [] if gap else [self.obstacle.cell]
        path = mg.astar(self.maze, cur, self.goal_cell, blocked)
        self.n_astar += 1
        if path is None or len(path) < 2 or (gap and self.obstacle.cell not in path[1:-1]):
            self.log(t, 'no_detour', cell=list(cur))
            return False
        if self.cfg['waypoint_mode'] == 'sparse':
            wp = mg.sparse_waypoints(self.maze, path, [self.obstacle.cell], self.cfg['max_leg_cells'])
        else:
            wp = mg.waypoints(self.maze, path, blocked, self.cfg['max_leg_cells'],
                              corners=self.cfg['waypoint_mode'] == 'corners')
        self.legs = [dict(cell=path[i], xy=mu.cell_center(path[i]), radius=self.cfg['waypoint_radius'])
                     for i in wp]
        if gap:
            self.add_gap_waypoints(path, wp)
        self.legs[-1]['xy'] = self.target
        self.leg_i, self.route_blocked = 0, blocked
        self.log(t, 'astar', route_cells=len(path) - 1, through_gap=gap,
                 waypoints=[list(leg['cell']) for leg in self.legs])
        return True

    def add_gap_waypoints(self, path, wp):
        """Make the cells before and after the obstacle cell waypoints, moved sideways onto the centre line
        of the gap (as the oracle's route does), so the leg through the obstacle cell runs straight along
        the gap. The ball has to come within gap_waypoint_radius of them: the gap leaves little room."""
        i = path.index(self.obstacle.cell)
        gap = self.obstacle.gap_center()
        w = int(np.argmax(np.abs(gap - mu.cell_center(self.obstacle.cell))))     # axis across the corridor
        idx = sorted(set(wp) | {i - 1, i + 1})
        legs = []
        for j in idx:
            xy = mu.cell_center(path[j])
            radius = self.cfg['waypoint_radius']
            if j in (i - 1, i + 1):
                xy[w], radius = gap[w], self.cfg['gap_waypoint_radius']
            legs.append(dict(cell=path[j], xy=xy, radius=radius))
        self.legs = legs

    def plan_leg(self, t, state, why):
        leg = self.legs[self.leg_i]
        path = mg.astar(self.maze, mu.xy_to_cell(state[:2]), leg['cell'], self.route_blocked)
        cells = max(1, len(path) - 1) if path else 1
        h = self.planner.horizon_for(cells, self.cfg['steps_per_cell'], self.cfg['min_horizon'])
        if self.method == 'astar_line':
            return self.line_leg(t, state, leg, path, cells, h, why)
        n = self.cfg.get('n_samples_leg') or self.cfg['n_samples']
        if self.cfg['leg_start_at_rest']:
            # Condition on the position only. Conditioned on its real velocity, a ball that enters a
            # corner waypoint at speed gets plans that carry on in the old direction.
            state = np.concatenate([state[:2], np.zeros(2)])
        n_clear, n_on_route = self.sample(t, state, leg['xy'], n, h, True, route=path)
        self.log(t, 'leg_plan', why=why, leg=self.leg_i, waypoint=list(leg['cell']), cells=cells, horizon=h,
                 n=n, n_clear=n_clear, n_on_route=n_on_route)
        return n_clear

    def line_leg(self, t, state, leg, path, cells, h, why):
        """astar_line: the leg is a constant-speed line from the ball through the turns of its A* route to
        the waypoint (a single straight line with waypoint_mode corners), with the same horizon as ours.
        With line_turns false the turns are dropped: one straight line to the waypoint, whatever lies between."""
        use_turns = path and self.cfg.get('line_turns', True)
        turns = [mu.cell_center(path[i]) for i in range(1, len(path) - 1) if mg.is_corner(path, i)] if use_turns else []
        plan = mg.line_plan([state[:2], *turns, leg['xy']], h, self.env.unwrapped.dt)
        n_clear = int(ob.first_collision(plan[:, :2], self.obstacle, self.cfg['check_margin']) is None)
        self.seq, self.t0 = plan, t
        self.plans.append((t, plan))
        self.log(t, 'leg_plan', why=why, leg=self.leg_i, waypoint=list(leg['cell']), cells=cells, horizon=h,
                 n=0, n_clear=n_clear, n_on_route=None)
        return n_clear

    def advance_legs(self, t, state):
        """Move on when the waypoint is reached; plan the same leg again when its plan has run out."""
        leg = self.legs[self.leg_i]
        last = self.leg_i == len(self.legs) - 1
        if not last and np.linalg.norm(state[:2] - leg['xy']) <= leg['radius']:
            self.log(t, 'waypoint_reached', leg=self.leg_i, waypoint=list(leg['cell']))
            self.leg_i += 1
            self.plan_leg(t, state, 'next_leg')
        elif t - self.t0 >= len(self.seq) - 1 + self.cfg['leg_grace_steps']:
            self.plan_leg(t, state, 'leg_ran_out')

    # ---- per-step decision ---------------------------------------------------------------------

    def replan_to_goal(self, t, state, use_obstacle):
        n = self.cfg['n_samples'] if use_obstacle else 1
        n_clear, _ = self.sample(t, state, self.target, n, self.goal_horizon(t), use_obstacle)
        self.log(t, 'replan', horizon=len(self.seq), n=n, n_clear=n_clear)
        return n_clear

    def before_step(self, t, state):
        cfg, method = self.cfg, self.method
        visible = t >= self.obstacle.t_appear
        if method == 'B0':
            return
        if method == 'oracle':
            if t == 0 or t == self.obstacle.t_appear:
                self.oracle_route(t, state, visible)
            return
        if method == 'B1':
            if t > 0 and t % cfg['k_fixed'] == 0:
                self.replan_to_goal(t, state, visible)
            return

        # B2 / ours: predicted-collision trigger
        if self.legs is not None:
            self.advance_legs(t, state)
        if not visible or (t - self.obstacle.t_appear) % cfg['k'] != 0:
            return
        if not ob.collides(self.seq[max(t - self.t0, 0):, :2], self.obstacle, cfg['check_margin']):
            return
        self.n_triggers += 1
        if self.n_triggers > cfg['max_triggers']:
            return
        self.log(t, 'trigger', leg=self.leg_i if self.legs is not None else None)
        if self.legs is not None:
            if self.plan_leg(t, state, 'leg_hits_obstacle') == 0 and self.start_route(t, state):
                self.plan_leg(t, state, 'rerouted')
            return
        if self.reroutes and self.needs_gap(state):
            # A rare plan to the goal that clears a narrow gap fails on tracking error (probe_partial)
            if self.start_route(t, state):
                self.plan_leg(t, state, 'first_leg')
            return
        if self.replan_to_goal(t, state, True) > 0:
            return
        if self.reroutes and self.start_route(t, state):
            self.plan_leg(t, state, 'first_leg')

    def action(self, t, state):
        if self.method == 'oracle':
            return self.oracle_action(state)
        return waypoint_action(self.seq, t - self.t0, state)

    # ---- oracle --------------------------------------------------------------------------------

    def oracle_route(self, t, state, visible):
        full = visible and self.obstacle.kind == 'full'
        path = mg.astar(self.maze, mu.xy_to_cell(state[:2]), self.goal_cell, [self.obstacle.cell] if full else [])
        if path is None:
            self.log(t, 'no_detour')
            return
        pts = [mu.cell_center(c) for c in path]
        pts[-1] = self.target
        if visible and not full and self.obstacle.cell in path:
            i = path.index(self.obstacle.cell)
            gap = self.obstacle.gap_center()
            w = int(np.argmax(np.abs(gap - mu.cell_center(self.obstacle.cell))))     # axis across the corridor
            for j in range(max(i - 1, 0), min(i + 1, len(pts) - 2) + 1):             # never moves the goal
                pts[j][w] = gap[w]
        self.path_xy = np.array(pts)
        self.log(t, 'astar', route_cells=len(path) - 1)

    def oracle_action(self, state):
        """Chase the route point after the nearest one with a speed-limited PD controller."""
        i = int(np.argmin(np.linalg.norm(self.path_xy - state[:2], axis=1)))
        aim = self.path_xy[min(i + 1, len(self.path_xy) - 1)]
        v_des = self.cfg['oracle_kp'] * (aim - state[:2])
        speed = np.linalg.norm(v_des)
        if speed > self.cfg['oracle_vmax']:
            v_des *= self.cfg['oracle_vmax'] / speed
        return self.cfg['oracle_kv'] * (v_des - state[2:])


def classify_failure(rec, loop, stalled, late_contact):
    """Rule-based failure label (empty string for successes). Order matters."""
    if rec['success']:
        return ''
    if rec['excluded_reason']:
        return 'excluded'
    if rec['obstacle_collision']:
        return 'obstacle_collision'
    if stalled and late_contact:
        return 'stuck_on_wall'
    if stalled:
        return 'stalled_off_goal'
    return 'timeout_still_moving'


def run_episode(sc, plan0, method, env, planner, cfg):
    maze = env.maze_arr
    seed = sc['seed']
    env.seed(seed)
    env.reset()
    env.set_state(np.array(sc['qpos']), np.array(sc['qvel']))
    set_all_seeds(cfg['replan_seed_offset'] + seed)         # same sampling noise for every method's replans
    target = np.array(sc.get('target', env._target), dtype=np.float64)     # random_goal scenarios store theirs
    obstacle = ob.Obstacle.from_dict(sc['obstacle'])
    loop = PlanningLoop(method, env, planner, cfg, obstacle, plan0, target)
    radius = cfg['success_radius']

    t_ep0 = time.perf_counter()
    rollout_xy = [env.state_vector()[:2].copy()]
    contact = []
    success_t = collision_t = None
    excluded = ''
    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()
        if t == obstacle.t_appear and ob.ball_hits(state[:2], obstacle):
            excluded = 'obstacle_appeared_on_agent'
            break
        loop.before_step(t, state)
        obs, _, _, _ = env.step(loop.action(t, state))
        xy = obs[:2].copy()
        rollout_xy.append(xy)
        contact.append(mu.wall_contact(env))
        if t + 1 > obstacle.t_appear and ob.ball_hits(xy, obstacle):     # the state at t_appear is checked above
            collision_t = t + 1
            break
        if np.linalg.norm(xy - target) <= radius:
            success_t = t + 1
            break
    episode_time = time.perf_counter() - t_ep0

    rollout_xy = np.array(rollout_xy)
    contact = np.array(contact, dtype=bool)
    dists = np.linalg.norm(rollout_xy - target, axis=1)
    stalled = len(rollout_xy) > 101 and mu.path_length(rollout_xy[-101:]) < 0.1
    late_contact = bool(contact[-100:].any())
    exec_cells = mu.cell_sequence(rollout_xy)
    replans = [e for e in loop.events if e['event'] in ('replan', 'leg_plan')]
    first = next((e for e in loop.events if e['event'] == 'replan' and e['n'] > 1), None)
    o = sc['obstacle']
    rec = dict(
        method=method, env=env.name, seed=seed, obstacle=o['kind'],
        blocked_cell=f"{o['cell'][0]},{o['cell'][1]}", t_appear=o['t_appear'],
        start_cell=f"{sc['start_cell'][0]},{sc['start_cell'][1]}",
        goal_cell=f"{sc['goal_cell'][0]},{sc['goal_cell'][1]}",
        base_cells=sc['base_cells'], detour_cells=sc['detour_cells'],
        # outcome
        success=success_t is not None, first_success_step=success_t,
        obstacle_collision=collision_t is not None, obstacle_collision_step=collision_t,
        obstacle_collisions=int(collision_t is not None),
        excluded_reason=excluded,
        final_dist_to_goal=float(dists[-1]), min_dist_to_goal=float(dists.min()),
        steps=len(rollout_xy) - 1,
        # cost (the stored t=0 plan counts as one plan for the Diffuser methods)
        planning_time_s=float(loop.planning_time + (0.0 if method == 'oracle' else sc['plan0_time_s'])),
        n_plans=loop.n_calls + (0 if method == 'oracle' else 1),
        n_samples=loop.n_samples + (0 if method == 'oracle' else 1),
        n_replans=len(replans), n_triggers=loop.n_triggers, n_astar_calls=loop.n_astar,
        first_replan_n_clear=first['n_clear'] if first else None,
        used_astar=bool(loop.n_astar) and loop.reroutes,
        n_no_detour=sum(e['event'] == 'no_detour' for e in loop.events),
        legs=len(loop.legs) if loop.legs is not None else 0,
        waypoints_reached=sum(e['event'] == 'waypoint_reached' for e in loop.events),
        episode_time_s=float(episode_time),
        # path and contacts
        path_length=mu.path_length(rollout_xy),
        exec_cells_visited=len(exec_cells),
        wall_contact_steps=int(contact.sum()), wall_contact_events=mu.contact_events(contact),
        adopted_plans_with_wall_points=sum(mu.count_points_in_wall(maze, p[:, :2]) > 0 for _, p in loop.plans),
        stalled_last100=bool(stalled),
    )
    rec['failure_reason'] = classify_failure(rec, loop, stalled, late_contact)
    rec['events'] = loop.events
    traj = dict(rollout_xy=rollout_xy, contact=contact, exec_cells=np.array(exec_cells),
                plan_starts=np.array([t for t, _ in loop.plans]),
                **{f'plan{i}': p for i, (_, p) in enumerate(loop.plans)})
    return rec, traj, loop


def render_episode(path, maze, rec, loop, rollout_xy, target, radius):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mu.draw_maze(ax, maze)
    ob.draw_obstacle(ax, loop.obstacle)
    for i, (t, p) in enumerate(loop.plans):
        ax.plot(p[:, 1], p[:, 0], '-', c='tab:blue', lw=0.8, alpha=0.25 if i else 0.6,
                label='plans' if i == 0 else None)
    if loop.path_xy is not None:
        ax.plot(loop.path_xy[:, 1], loop.path_xy[:, 0], ':', c='tab:blue', lw=1.0, label='A* route')
    if loop.legs is not None:
        w = np.array([leg['xy'] for leg in loop.legs])
        ax.scatter(w[:, 1], w[:, 0], marker='x', c='k', s=30, zorder=6, label='waypoints')
    r = rollout_xy
    ax.plot(r[:, 1], r[:, 0], '-', c='tab:orange', lw=1.4, label='executed')
    k = min(loop.obstacle.t_appear, len(r) - 1)
    ax.scatter([r[k, 1]], [r[k, 0]], marker='v', c='tab:red', s=30, zorder=6, label='agent when obstacle appears')
    ax.scatter([r[0, 1]], [r[0, 0]], marker='o', c='green', s=40, zorder=6, label='start')
    ax.add_patch(plt.Circle((target[1], target[0]), radius, fill=False, ec='magenta', lw=1.5))
    status = 'SUCCESS' if rec['success'] else f"FAIL ({rec['failure_reason']})"
    ax.set_title(f"{rec['method']} seed {rec['seed']} ({rec['obstacle']} at {rec['blocked_cell']}, "
                 f"t={rec['t_appear']}): {status}\nsteps={rec['steps']}, replans={rec['n_replans']}, "
                 f"samples={rec['n_samples']}, plan {rec['planning_time_s']:.1f}s", fontsize=8)
    ax.legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--device')
    ap.add_argument('--no_png', action='store_true')
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if args.device:
        cfg['device'] = args.device
    base = os.path.join(REPO, cfg['out'])
    scen = json.load(open(os.path.join(base, 'scenarios.json')))['scenarios']
    plans0 = np.load(os.path.join(base, 'plans0.npz'))
    if args.seeds:
        scen = [s for s in scen if s['seed'] in args.seeds]
    out = os.path.join(base, args.method)
    os.makedirs(os.path.join(out, 'episodes'), exist_ok=True)
    save_png = cfg.get('save_png', True) and not args.no_png
    if save_png:
        import matplotlib
        matplotlib.use('Agg')

    t0 = time.perf_counter()
    planner = FrozenDiffuserPlanner(cfg['env'], cfg['diffusion_loadpath'], cfg['diffusion_epoch'],
                                    cfg['device'], threads=cfg.get('torch_threads', 0))
    env = load_env(cfg['env'])
    meta = dict(config=cfg, config_path=os.path.relpath(os.path.abspath(args.config), REPO), method=args.method,
                seeds=[s['seed'] for s in scen], git_commit=git_commit(), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                python=sys.version.split()[0], max_episode_steps=env.max_episode_steps,
                target=list(map(float, env._target)), **planner.meta())
    with open(os.path.join(out, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[{args.method}] model epoch {planner.epoch} on {planner.device} | {len(scen)} scenarios', flush=True)

    for sc in scen:
        seed = sc['seed']
        rec, traj, loop = run_episode(sc, plans0[f'seed{seed:03d}'], args.method, env, planner, cfg)
        stem = os.path.join(out, 'episodes', f'seed{seed:03d}')
        with open(stem + '.json', 'w') as f:
            json.dump(rec, f, indent=2)
        np.savez_compressed(stem + '.npz', **traj)
        if save_png:
            render_episode(stem + '.png', env.maze_arr, rec, loop, traj['rollout_xy'], env._target,
                           cfg['success_radius'])
        print(f"[{args.method}] seed {seed:3d} | {rec['obstacle']:7s} at {rec['blocked_cell']:>5} | "
              f"success={rec['success']!s:5} | steps={rec['steps']:3d} | replans={rec['n_replans']:2d} "
              f"| samples={rec['n_samples']:3d} | plan {rec['planning_time_s']:.1f}s | {rec['failure_reason']}",
              flush=True)
    print(f'[{args.method}] done: {len(scen)} episodes in {time.perf_counter() - t0:.1f}s -> '
          f'{os.path.relpath(out, REPO)}', flush=True)


if __name__ == '__main__':
    main()
