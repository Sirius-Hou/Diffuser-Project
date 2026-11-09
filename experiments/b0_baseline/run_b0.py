"""
B0 batch evaluation: frozen pretrained Diffuser on maze2d-large-v1, plan ONCE at t=0, no replanning.

Same control loop as scripts/plan_maze2d.py (same Policy call, goal inpainting at H-1,
waypoint controller), repeated over seeded episodes, recording the proposal metrics.
Step 1 runs without obstacles; obstacle fields are kept (= 0) so later runs share the schema.

    bash experiments/common/env/run.sh python experiments/b0_baseline/run_b0.py --config experiments/b0_baseline/configs/smoke.json
    bash experiments/common/env/run.sh python experiments/b0_baseline/run_b0.py --config experiments/b0_baseline/configs/main.json
Options override the config: --seeds 3 4  --out DIR  --tag _rep  --no_png
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common.planner import FrozenDiffuserPlanner, load_env, set_all_seeds, waypoint_action  # noqa: E402


def git_commit():
    try:
        sha = subprocess.check_output(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.call(['git', '-C', REPO, 'diff', '--quiet', 'HEAD', '--', 'diffuser', 'scripts', 'config']) != 0
        return sha + ('+dirty-core' if dirty else '')
    except Exception:  # noqa: BLE001
        return 'unknown'


def classify_failure(success, plan_wall_pts, stalled, late_contact):
    """Rule-based failure label (empty string for successes). Order matters."""
    if success:
        return ''
    if plan_wall_pts > 0:
        return 'plan_through_wall'        # the sampled plan itself passes through a wall
    if stalled and late_contact:
        return 'stuck_on_wall'            # agent pinned against a wall for the last 100 steps
    if stalled:
        return 'stalled_off_goal'         # agent stopped away from the goal, no wall contact
    return 'timeout_still_moving'         # still moving at the step limit


def run_episode(seed, env, planner, success_radius):
    maze = env.maze_arr
    set_all_seeds(seed)
    env.seed(seed)
    t_ep0 = time.perf_counter()
    obs = env.reset()
    target = np.array(env._target, dtype=np.float64)
    goal_cell = mu.xy_to_cell(target)
    start_xy = obs[:2].copy()
    start_cell = mu.xy_to_cell(start_xy)
    shortest = mu.bfs_path(maze, start_cell, goal_cell)
    shortest_len = len(shortest) - 1 if shortest else None

    # ---- B0: a single plan at t = 0
    plans, _, planning_time = planner.plan(obs, target, n=1)
    sequence = plans[0]                                 # (H, 4) [x, y, vx, vy]
    plan_xy = sequence[:, :2]
    H = len(plan_xy)

    # ---- open-loop execution with the original controller
    rollout_xy = [obs[:2].copy()]
    total_reward = 0.0
    first_success_t = None
    contact, track_err = [], []
    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()
        action = waypoint_action(sequence, t, state)
        obs, reward, terminal, _ = env.step(action)
        total_reward += reward
        xy = obs[:2].copy()
        rollout_xy.append(xy)
        contact.append(mu.wall_contact(env))
        track_err.append(float(np.linalg.norm(xy - plan_xy[min(t + 1, H - 1)])))
        if first_success_t is None and np.linalg.norm(xy - target) <= success_radius:
            first_success_t = t + 1
        if terminal:
            break
    episode_time = time.perf_counter() - t_ep0

    rollout_xy = np.array(rollout_xy)
    contact = np.array(contact, dtype=bool)
    dists = np.linalg.norm(rollout_xy - target, axis=1)
    success = first_success_t is not None
    final_dist = float(dists[-1])

    plan_wall_pts = mu.count_points_in_wall(maze, plan_xy)
    plan_wall_touch = mu.count_points_in_wall(maze, plan_xy, mu.BALL_RADIUS)
    plan_cells = mu.cell_sequence(plan_xy)
    exec_cells = mu.cell_sequence(rollout_xy)
    plan_reach = np.where(np.linalg.norm(plan_xy - target, axis=1) <= success_radius)[0]

    stalled = mu.path_length(rollout_xy[-101:]) < 0.1
    late_contact = bool(contact[-100:].any())
    seg = np.linalg.norm(np.diff(rollout_xy, axis=0), axis=1)
    remaining = np.cumsum(seg[::-1])[::-1]
    idx = np.where(remaining < 0.1)[0]
    stall_start = int(idx[0]) if (len(idx) and not success) else None

    rec = dict(
        method='B0', env=env.name, seed=seed, obstacle='none',
        start_x=float(start_xy[0]), start_y=float(start_xy[1]),
        start_cell=f'{start_cell[0]},{start_cell[1]}', goal_cell=f'{goal_cell[0]},{goal_cell[1]}',
        start_goal_euclid=float(np.linalg.norm(start_xy - target)),
        shortest_path_cells=shortest_len,
        # outcome
        success=bool(success),
        first_success_step=first_success_t,
        final_dist_to_goal=final_dist,
        min_dist_to_goal=float(dists.min()),
        success_at_end=bool(final_dist <= success_radius),
        reached_then_left=bool(success and final_dist > success_radius),
        steps=len(rollout_xy) - 1,
        d4rl_return=float(total_reward),
        d4rl_normalized_score=float(100 * env.get_normalized_score(total_reward)),
        # cost
        planning_time_s=float(planning_time), n_plans=1, n_replans=0,
        episode_time_s=float(episode_time),
        # path
        path_length=mu.path_length(rollout_xy),
        path_length_to_success=mu.path_length(rollout_xy[:first_success_t + 1]) if success else None,
        plan_path_length=mu.path_length(plan_xy),
        exec_cells_visited=len(exec_cells),
        plan_cells_visited=len(plan_cells),
        plan_detour_cells=(len(plan_cells) - 1 - shortest_len) if shortest_len is not None else None,
        # contacts
        wall_contact_steps=int(contact.sum()),
        wall_contact_events=mu.contact_events(contact),
        obstacle_collisions=0,
        # plan quality / diagnostics
        plan_end_dist_to_goal=float(np.linalg.norm(plan_xy[-1] - target)),
        plan_first_reach_idx=int(plan_reach[0]) if len(plan_reach) else None,
        plan_wall_points=plan_wall_pts,
        plan_wall_touch_points=plan_wall_touch,
        plan_max_step_jump=float(np.linalg.norm(np.diff(plan_xy, axis=0), axis=1).max()),
        track_err_mean=float(np.mean(track_err)),
        track_err_max=float(np.max(track_err)),
        track_err_mean_first_H=float(np.mean(track_err[:H - 1])),
        dist_to_goal_at_plan_end=float(dists[min(H - 1, len(dists) - 1)]),
        stall_start_step=stall_start,
        stalled_last100=bool(stalled),
        failure_reason=classify_failure(success, plan_wall_pts, stalled, late_contact),
    )
    traj = dict(plan=sequence, rollout_xy=rollout_xy, contact=contact,
                plan_cells=np.array(plan_cells), exec_cells=np.array(exec_cells),
                shortest_cells=np.array(shortest) if shortest else np.zeros((0, 2)))
    return rec, traj


def render_episode(path, maze, rec, traj, target, success_radius):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mu.draw_maze(ax, maze)
    p, r, c = traj['plan'][:, :2], traj['rollout_xy'], traj['contact']
    ax.plot(p[:, 1], p[:, 0], '-', c='tab:blue', lw=1.2, label='plan (t=0)')
    ax.plot(r[:, 1], r[:, 0], '-', c='tab:orange', lw=1.2, label='executed')
    if c.any():
        cr = r[1:][c]
        ax.scatter(cr[:, 1], cr[:, 0], s=6, c='red', zorder=5, label='wall contact')
    ax.scatter([r[0, 1]], [r[0, 0]], marker='o', c='green', s=40, zorder=6, label='start')
    ax.add_patch(plt.Circle((target[1], target[0]), success_radius, fill=False, ec='magenta', lw=1.5))
    status = 'SUCCESS' if rec['success'] else f"FAIL ({rec['failure_reason']})"
    ax.set_title(f"B0 seed {rec['seed']}: {status}\nfinal d={rec['final_dist_to_goal']:.2f}, "
                 f"contacts={rec['wall_contact_events']}, plan {rec['planning_time_s']:.1f}s", fontsize=9)
    ax.legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--out')
    ap.add_argument('--tag', default='', help='suffix for per-episode files (e.g. _rep for repro runs)')
    ap.add_argument('--no_png', action='store_true')
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    assert cfg.get('obstacle') in (None, 'none'), 'B0 step 1 runs without obstacles'
    seeds = args.seeds or cfg.get('seeds') or list(range(cfg['seed_start'], cfg['seed_start'] + cfg['n_episodes']))
    out = os.path.join(REPO, args.out or cfg['out'])
    os.makedirs(os.path.join(out, 'episodes'), exist_ok=True)
    save_png = cfg.get('save_png', True) and not args.no_png
    if save_png:
        import matplotlib
        matplotlib.use('Agg')

    t0 = time.perf_counter()
    planner = FrozenDiffuserPlanner(cfg['env'], cfg['diffusion_loadpath'], cfg['diffusion_epoch'],
                                    cfg['device'], threads=cfg.get('torch_threads', 0))
    env = load_env(cfg['env'])
    print(f"[b0] model epoch {planner.epoch} (H={planner.horizon}, T={planner.n_diffusion_steps}) loaded in "
          f"{planner.load_time_s:.1f}s | seeds={seeds}", flush=True)

    meta = dict(config=cfg, config_path=os.path.relpath(os.path.abspath(args.config), REPO), seeds=seeds,
                git_commit=git_commit(), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                python=sys.version.split()[0], max_episode_steps=env.max_episode_steps,
                target=list(map(float, env._target)), **planner.meta())
    with open(os.path.join(out, f'run_meta{args.tag}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    for seed in seeds:
        rec, traj = run_episode(seed, env, planner, cfg['success_radius'])
        rec['diffusion_epoch'] = planner.epoch
        stem = os.path.join(out, 'episodes', f'seed{seed:03d}{args.tag}')
        with open(stem + '.json', 'w') as f:
            json.dump(rec, f, indent=2)
        np.savez_compressed(stem + '.npz', **traj)
        if save_png:
            render_episode(stem + '.png', env.maze_arr, rec, traj, env._target, cfg['success_radius'])
        print(f"[b0] seed {seed:3d} | start {rec['start_cell']:>5} | success={rec['success']!s:5} "
              f"| final_d={rec['final_dist_to_goal']:.3f} | plan {rec['planning_time_s']:.2f}s "
              f"| contacts={rec['wall_contact_events']} | {rec['failure_reason']}", flush=True)
    print(f'[b0] done: {len(seeds)} episodes in {time.perf_counter() - t0:.1f}s -> {os.path.relpath(out, REPO)}',
          flush=True)


if __name__ == '__main__':
    main()
