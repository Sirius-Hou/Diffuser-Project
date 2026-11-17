"""
Blocked-corridor evaluation: one planning loop per method around the same frozen Diffuser.

Every method starts from the stored state and t=0 plan of each scenario (see make_scenarios.py).
The obstacle becomes known at t_appear; touching it ends the episode as a failure.

  B0     : never replans
  B1     : replans every `k_fixed` steps, best-of-N against the obstacle once it is known
  B2     : every `k` steps checks the rest of the plan against the obstacle; replans (best-of-N) only on a hit
  ours   : B2, and when all N samples hit: A* around the blocked cell, then one short plan per waypoint
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

METHODS = ('B0', 'B1', 'B2', 'ours', 'oracle')


class PlanningLoop:
    """Decides, before every control step, whether to replace the current plan."""

    def __init__(self, method, env, planner, cfg, obstacle, plan0, target):
        self.method, self.env, self.planner, self.cfg = method, env, planner, cfg
        self.maze, self.obstacle, self.target = env.maze_arr, obstacle, target
        self.goal_cell = mu.xy_to_cell(target)
        self.seq, self.t0 = plan0, 0                # current plan and the step it started at
        self.legs, self.leg_i = None, 0             # ours: waypoint legs once A* has been used
        self.path_xy = None                         # oracle: route as a list of points
        self.plans = [] if method == 'oracle' else [(0, plan0)]     # every adopted plan, for plotting
        self.events = []
        self.planning_time = 0.0
        self.n_calls = self.n_samples = self.n_triggers = self.n_astar = 0

    def log(self, t, event, **info):
        self.events.append(dict(t=int(t), event=event, **info))

    # ---- sampling and selection ---------------------------------------------------------------

    def goal_horizon(self, t):
        """Horizon for a plan to the goal. 'remaining' shortens it so the plan ends before the step limit."""
        H, m = self.planner.horizon, self.planner.horizon_multiple
        if self.cfg['replan_horizon'] == 'full':
            return H
        left = self.env.max_episode_steps - t - self.cfg['budget_slack_steps']
        return int(min(H, max(self.cfg['min_horizon'], m * (left // m))))

    def sample(self, t, state, goal_xy, n, horizon, use_obstacle):
        """Sample n plans and adopt the first one clear of the obstacle (else the one that hits it latest)."""
        plans, _, dt = self.planner.plan(state, goal_xy, n=n, horizon=horizon)
        self.planning_time += dt
        self.n_calls += 1
        self.n_samples += n
        if use_obstacle:
            hits = [ob.first_collision(p[:, :2], self.obstacle, self.cfg['check_margin']) for p in plans]
            clear = [i for i, h in enumerate(hits) if h is None]
            best = clear[0] if clear else int(np.argmax(hits))
        else:
            clear, best = list(range(n)), 0
        self.seq, self.t0 = plans[best], t
        self.plans.append((t, plans[best]))
        return len(clear)

    # ---- ours: A* route and waypoint legs ------------------------------------------------------

    def start_route(self, t, state):
        cur = mu.xy_to_cell(state[:2])
        blocked = [self.obstacle.cell]
        path = mg.astar(self.maze, cur, self.goal_cell, blocked)
        self.n_astar += 1
        if path is None or len(path) < 2:
            self.log(t, 'no_detour', cell=list(cur))
            return False
        wp = mg.waypoints(self.maze, path, blocked, self.cfg['max_leg_cells'])
        self.legs = [dict(cell=path[i], xy=mu.cell_center(path[i])) for i in wp]
        self.legs[-1]['xy'] = self.target
        self.leg_i = 0
        self.log(t, 'astar', route_cells=len(path) - 1, waypoints=[list(path[i]) for i in wp])
        return True

    def plan_leg(self, t, state, why):
        leg = self.legs[self.leg_i]
        path = mg.astar(self.maze, mu.xy_to_cell(state[:2]), leg['cell'], [self.obstacle.cell])
        cells = max(1, len(path) - 1) if path else 1
        h = self.planner.horizon_for(cells, self.cfg['steps_per_cell'], self.cfg['min_horizon'])
        n_clear = self.sample(t, state, leg['xy'], self.cfg['n_samples'], h, True)
        self.log(t, 'leg_plan', why=why, leg=self.leg_i, waypoint=list(leg['cell']), cells=cells, horizon=h,
                 n_clear=n_clear)
        return n_clear

    def advance_legs(self, t, state):
        """Move on when the waypoint is reached; plan the same leg again when its plan has run out."""
        leg = self.legs[self.leg_i]
        last = self.leg_i == len(self.legs) - 1
        if not last and np.linalg.norm(state[:2] - leg['xy']) <= self.cfg['waypoint_radius']:
            self.log(t, 'waypoint_reached', leg=self.leg_i, waypoint=list(leg['cell']))
            self.leg_i += 1
            self.plan_leg(t, state, 'next_leg')
        elif t - self.t0 >= len(self.seq) - 1 + self.cfg['leg_grace_steps']:
            self.plan_leg(t, state, 'leg_ran_out')

    # ---- per-step decision ---------------------------------------------------------------------

    def replan_to_goal(self, t, state, use_obstacle):
        n = self.cfg['n_samples'] if use_obstacle else 1
        n_clear = self.sample(t, state, self.target, n, self.goal_horizon(t), use_obstacle)
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
        if self.replan_to_goal(t, state, True) > 0:
            return
        if method == 'ours' and self.start_route(t, state):
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
        if visible and not full and self.obstacle.cell in path:
            pts[path.index(self.obstacle.cell)] = self.obstacle.gap_center()
        pts[-1] = self.target
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
    if any(e['event'] == 'no_detour' for e in loop.events):
        return 'no_detour'
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
    target = np.array(env._target, dtype=np.float64)
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
        if t + 1 >= obstacle.t_appear and ob.ball_hits(xy, obstacle):
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
        used_astar=bool(loop.n_astar) and method == 'ours',
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
